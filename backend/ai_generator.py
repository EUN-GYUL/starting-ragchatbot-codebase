import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content** - For questions about specific course content, concepts, or detailed educational materials
2. **get_course_outline** - For questions about course structure, syllabus, lesson listings, or course overviews

Tool Usage Guidelines:
- **Course outline/syllabus queries**: Use get_course_outline to retrieve course title, course link, and complete lesson listings
- **Course content queries**: Use search_course_content for specific topics, concepts, or detailed materials
- **Sequential tool usage**: You may use up to 2 rounds of tool calls when needed for complex queries
- Use results from first tool call to inform second tool call if additional information is needed
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool, then provide the course title, course link, and numbered lesson list
- **Course content questions**: Use search_course_content tool, then answer based on results
- **Complex queries**: Use sequential tool calls when you need information from multiple sources or need to refine your search
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results" or similar phrases


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle sequential execution of tool calls with up to 2 rounds.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        MAX_ROUNDS = 2
        
        # Initialize tracking variables
        current_response = initial_response
        messages = base_params["messages"].copy()
        round_count = 0
        
        # Tool execution loop
        while (current_response.stop_reason == "tool_use" and 
               round_count < MAX_ROUNDS and 
               tool_manager):
            
            round_count += 1
            
            # Add AI's tool use response to messages
            messages.append({"role": "assistant", "content": current_response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name, 
                            **content_block.input
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}"
                        })
            
            # Add tool results as single message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            # Prepare next API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            
            # Include tools for potential further tool use (except on last round)
            if round_count < MAX_ROUNDS and "tools" in base_params:
                api_params["tools"] = base_params["tools"]
                api_params["tool_choice"] = {"type": "auto"}
            
            # Make next API call
            try:
                current_response = self.client.messages.create(**api_params)
            except Exception as e:
                # Handle API errors gracefully
                return f"Error in tool execution round {round_count}: {str(e)}"
        
        # Return final response text
        if hasattr(current_response, 'content') and current_response.content:
            return current_response.content[0].text
        return "No response generated"