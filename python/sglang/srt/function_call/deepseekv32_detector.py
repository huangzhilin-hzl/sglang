import json
import logging
import re
from typing import List, Literal, Optional, Union

from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _partial_json_loads

try:
    from xgrammar import StructuralTag
    from xgrammar.structural_tag import (
        AnyTextFormat,
        ConstStringFormat,
        JSONSchemaFormat,
        SequenceFormat,
        TagFormat,
        TagsWithSeparatorFormat,
        TriggeredTagsFormat,
    )
except ImportError:
    StructuralTag = None  # type: ignore

logger = logging.getLogger(__name__)

# Names mirror the DeepSeek-V3.2 official chat template tokens
# (see encoding_dsv32.TOOLS_SYSTEM_TEMPLATE).
_INVOKE_BEGIN_PREFIX = '<｜DSML｜invoke name="'
_INVOKE_BEGIN_SUFFIX = '">\n'
_THINK_TAG_END = "</think>"
_THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
_XML_STYLE = "deepseek_xml"


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    MAX_STREAMING_BUFFER_CHARS = 65536

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        self.partial_parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = (
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>"
        )
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.prefix_invoke_end_call = ["</", "｜DSML｜", "inv", "oke"]
        self.current_tool_id = -1
        self.invoke_start_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*(/?)>'
        )
        self.max_streaming_buffer_chars = self.MAX_STREAMING_BUFFER_CHARS

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text

    def _find_common_prefix_len(self, s1: str, s2: str) -> int:
        min_length = min(len(s1), len(s2))
        for i in range(min_length):
            if s1[i] != s2[i]:
                return i
        return min_length

    def _find_invoke(
        self, text: str, start: int = 0, allow_partial: bool = False
    ) -> tuple[str, str, bool, int, int] | None:
        """Find one DSML invoke block without running a regex over the body."""
        match = self.invoke_start_regex.search(text, start)
        if not match:
            return None

        func_name = match.group(1).strip()
        if match.group(2):
            return func_name, "", True, match.start(), match.end()

        close_start = text.find(self.invoke_end_token, match.end())
        if close_start == -1:
            if not allow_partial:
                return None
            return func_name, text[match.end() :], False, match.start(), len(text)

        invoke_end = close_start + len(self.invoke_end_token)
        return (
            func_name,
            text[match.end() : close_start],
            True,
            match.start(),
            invoke_end,
        )

    def _clear_completed_wrapper_if_only_tail(self) -> None:
        if self.eot_token not in self._buffer:
            return
        tail = self._buffer.replace(self.eot_token, "")
        if tail.strip() == "":
            self._buffer = ""

    def _reset_streaming_state(self) -> None:
        self._buffer = ""
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []

    def _abort_streaming_parse(self, reason: str) -> StreamingParseResult:
        logger.warning("Abort DeepSeek DSML tool-call streaming parse: %s", reason)
        self._reset_streaming_state()
        return StreamingParseResult(normal_text="", calls=[])

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix in case they are captured by param
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
            param_name = match.group(1)
            param_type = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()

            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining_content = invoke_content[last_match_end:]

            # Remove incomplete parameter_end_call prefix in case they are captured by param
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)

            # Match start of a parameter tag + value (potentially incomplete)
            # Regex: <tag name="..." string="...">VALUE... (no end tag)
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                if partial_match.group(2) == "true":
                    parameters[param_name] = param_value.strip()
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(
                            param_value, Allow.ALL
                        )[0]
                    except json.JSONDecodeError:
                        parameters[param_name] = param_value.strip()

        return json.dumps(parameters, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].removesuffix("\n\n") if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = re.search(
                self.function_calls_regex,
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            search_start = 0
            while invoke := self._find_invoke(function_calls_content, search_start):
                func_name, invoke_content, _, _, invoke_end = invoke
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": json.loads(func_args)}
                calls.extend(self.parse_base_json(match_result, tools))
                search_start = invoke_end

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks and argument streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)

        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        if (
            not self.has_tool_call(current_text)
            and not potentially_dsml
            and not ends_with_prefix
        ):
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        if (
            self.max_streaming_buffer_chars is not None
            and len(self._buffer) > self.max_streaming_buffer_chars
        ):
            return self._abort_streaming_parse(
                f"buffer exceeded {self.max_streaming_buffer_chars} chars"
            )

        all_calls: list[ToolCallItem] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke = self._find_invoke(current_text, allow_partial=True)
                if not invoke:
                    break

                func_name, invoke_content, is_tool_end, _, invoke_end = invoke

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure arrays are large enough for current tool
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # 1. Send tool name if not sent yet
                if not self.current_tool_name_sent:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                # 2. Parse current parameters (partial or complete)
                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )

                # 3. Calculate and send incremental arguments
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if is_tool_end:
                    # If complete, send everything remaining
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    # If partial, send stable prefix diff
                    if current_params != prev_params:
                        prefix_len = self._find_common_prefix_len(
                            current_params, prev_params
                        )
                        if prefix_len > sent_len:
                            argument_diff = current_params[sent_len:prefix_len]

                if argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update the stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_end:]
                    current_text = self._buffer  # Update for next iteration

                    # Move to next tool call
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found
            self._clear_completed_wrapper_if_only_tail()
            return StreamingParseResult(normal_text="", calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
    ) -> Optional["StructuralTag"]:
        """
        Build an xgrammar StructuralTag locally for DeepSeek-V3.2.

        Both layers — the outer `<｜DSML｜function_calls｜>...` wrapper and
        the inner `<｜DSML｜invoke>...</｜DSML｜invoke>` blocks — are encoded
        directly in the grammar with a single-newline join between
        consecutive invokes, matching DeepSeek-V3.2's official chat
        template. This avoids two layered defects that surfaced with the
        prior `xgrammar.get_model_structural_tag("deepseek_v3_2")` path:

        - the xgrammar builtin template (pre mlc-ai/xgrammar#638) forced
          a double-newline join, which deterministically collapsed
          parallel tool calls to one at greedy decoding.
        - falling back to the legacy structural tag (built from
          `structure_info()`) only constrains the inner invoke block;
          the outer wrapper is off-grammar and the model can skip it
          under `at_least_one=True`, leaving `detect_and_parse` with no
          `<｜DSML｜function_calls>` marker to anchor on.

        Returning a fully-formed StructuralTag from the detector keeps
        both fixes local to sglang and decoupled from the xgrammar
        release cadence.
        """
        if not tools or StructuralTag is None:
            return None

        # `INVOKE_END` and the empty separator together yield a single `\n`
        # between consecutive invokes — matching DeepSeek-V3.2's chat template
        # `"\n".join(invoke_blocks)`.
        function_calls_begin = self.bot_token + "\n"
        invoke_end = self.invoke_end_token + "\n"

        def _invoke_tag(tool: Tool) -> TagFormat:
            return TagFormat(
                begin=_INVOKE_BEGIN_PREFIX + tool.function.name + _INVOKE_BEGIN_SUFFIX,
                content=JSONSchemaFormat(
                    json_schema=tool.function.parameters or {},
                    style=_XML_STYLE,
                ),
                end=invoke_end,
            )

        if isinstance(tool_choice, ToolChoice):
            target = next(
                (t for t in tools if t.function.name == tool_choice.function.name),
                None,
            )
            if target is None:
                return None
            invoke_tags = [_invoke_tag(target)]
            is_required = True
        else:
            invoke_tags = [_invoke_tag(t) for t in tools]
            is_required = tool_choice == "required"

        inner_tool_calls = TagsWithSeparatorFormat(
            tags=invoke_tags, separator="", at_least_one=True
        )

        if is_required:
            suffix_tag = SequenceFormat(
                elements=[
                    ConstStringFormat(value=function_calls_begin),
                    inner_tool_calls,
                    ConstStringFormat(value=self.eot_token),
                ]
            )
        else:
            suffix_tag = TriggeredTagsFormat(
                triggers=[self.bot_token],
                tags=[
                    TagFormat(
                        begin=function_calls_begin,
                        content=inner_tool_calls,
                        end=self.eot_token,
                    )
                ],
                excludes=_THINK_EXCLUDE_TOKENS,
            )

        if not thinking_mode:
            return StructuralTag(format=suffix_tag)

        prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=_THINK_TAG_END)
        return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))
