# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import backoff
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    The ChatGPT class handles interactions with the OpenAI models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False
    ) -> None:
        """
        Initialize the ChatGPT instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'chatgpt'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for chatgpt, i.e. gpt-4, gpt-3.5-turbo, etc.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        self.stop: Union[str, List[str]] = self.config["stop"]
        # The account organization is the organization that is used for chatgpt.
        self.organization: str = self.config["organization"]
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION is not set")
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")
        # Initialize the OpenAI Client
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the OpenAI model.
        :rtype: Dict
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "system", "content": "You are a helpful assistant, expert in writing States of the Art (SOTA)"}, {"role": "user", "content": query}], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "system", "content": "You are a helpful assistant, expert in writing States of the Art (SOTA)"}, {"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in chatgpt: {e}, trying again with {next_try} samples. If the error is about Tokens per Day limit, do not worry, instead of using GPT4, the lower version GPT3.5-Turbo will be used instead."
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.respone_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        Send chat messages to the OpenAI model and retrieves the model's response.
        Implements backoff on OpenAI error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The OpenAI model's response.
        :rtype: ChatCompletion
        """
        try:            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=num_responses,
                stop=self.stop,
            )

            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            completion_tokens_k = float(self.completion_tokens) / 1000.0
            self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
            )
            self.logger.info(
                f"This is the response from chatgpt: {response}"
                f"\nThis is the cost of the response: {self.cost}"
            )
            return response
        ### The following code is added to the original class, to change the model from GPT4 to GPT3.5 in case the Tokens per Day (TPD) limit is reached.
        except OpenAIError as e:
            if "on tokens_usage_based per day" in e.message:
                self.logger.warning("Error in token ussage. This error corresponds to TPD")
                    #    f"Error in chatgpt: {e}, this is the new modified part by Kevin."
                    #)

                    # Handle rate limit exceeded error here
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=messages,
                    temperature=1.25,
                    max_tokens=5000,
                    n=num_responses,
                    stop=self.stop,
                )

                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
                prompt_tokens_k = float(self.prompt_tokens) / 1000.0
                completion_tokens_k = float(self.completion_tokens) / 1000.0
                self.cost = (
                    self.prompt_token_cost * prompt_tokens_k
                    + self.response_token_cost * completion_tokens_k
                )
                self.logger.info(
                    f"You have exceeded the max. number of tokens per day with GPT4, so the following response has been obtained with GPT3.5. This is the response from GPT3.5: {response}"
                    f"\nThis is the cost of the response: {self.cost}"
                )
                #print("This error corresponds to TPD")
                return response
            elif "on tokens_usage_based per min" in e.message:
                self.logger.warning("Error in token ussage. This error corresponds to TPM")
                time.sleep(64)
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=num_responses,
                    stop=self.stop,
                )
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
                prompt_tokens_k = float(self.prompt_tokens) / 1000.0
                completion_tokens_k = float(self.completion_tokens) / 1000.0
                self.cost = (
                    self.prompt_token_cost * prompt_tokens_k
                    + self.response_token_cost * completion_tokens_k
                )
                self.logger.info(
                    f"You have exceeded the max. number of tokens per min with the model you are using. Therefore, a break of a min has been imposed. After that min, the model has been called again. This is the response from the model: {response}"
                    f"\nThis is the cost of the response: {self.cost}"
                )   
                return response

            else:
                # For other OpenAI errors, raise the exception again
                self.logger.warning("This is not an error related to Token Limits. Check the error!")

                raise e    
            

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the OpenAI model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]