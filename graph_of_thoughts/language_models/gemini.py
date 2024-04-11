# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Kevin Monsalvez-Pozo and Jorge Ruiz

import backoff
import os
import random
import time
from typing import List, Dict, Union

from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, GenerationResponse

from .abstract_language_model import AbstractLanguageModel


class Gemini(AbstractLanguageModel):
    """
    The Gemini class handles interactions with the Gemini models using the provided configuration.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "gemini", cache: bool = False
    ) -> None:
        """
        Initialize the Gemini instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'gemini'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used to generate responses.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The top_k is the number of tokens with the largest probability that the model will consider when selecting tokens.
        self.top_k: int = self.config["top_k"]
        # The the top_p is the probability mass that the model will use to select tokens.
        self.top_p = self.config["top_p"]
        # The number of response variations to return. This numbers must be 1.
        self.candidate_count = self.config["candidate_count"]       
        # The maximum number of tokens to generate in the chat completion.
        self.max_output_tokens: int = self.config["max_output_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        # self.stop_sequences: Union[str, List[str]] = self.config["stop_sequences"]
        # The project is the project created in the Google Cloud Platform
        self.project: str = self.config["project"]
        if self.project == "":
            self.logger.warning("The project is not set")
        # The location is the location established for the project in the Google Cloud Platform
        self.location: str = self.config["location"]
        if self.location == "":
            raise ValueError("The location is not set")
        # Initialize the Vertex AI client
        aiplatform.init(project = self.project, location = self.location)

    # KMP: The gemini model accepts as input images, those could be considered as arguments to query.
    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[GenerationResponse], GenerationResponse]:
        """
        Query the OpenAI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the Gemini model.
        :rtype: Dict
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        if num_responses == 1:
            response = self.chat([{ 'role': "user", 'parts': [{'text': query}] }])
        else:
            response = []
            total_num_attempts = num_responses 
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    res = self.chat([{ 'role': "user", 'parts': [{'text': query}] }])
                    response.append(res)
                    num_responses -= 1
                except Exception as e:
                    self.logger.warning(
                        f"Error in gemini: {e}, trying again with another sample."
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.respone_cache[query] = response

        return response

    @backoff.on_exception(backoff.expo, Exception, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict]) -> GenerationResponse:
        """
        Send chat messages to the Gemini model and retrieves the model's response.
        Implements backoff on Gemini error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :return: The Gemini model's response.
        :rtype: GenerationResponse
        """
        gemini_model = GenerativeModel(self.model_id)
        
        try:
            while True:            
                response = gemini_model.generate_content(
                    contents=messages,
                    generation_config= {
                        "temperature": self.temperature,
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "candidate_count": self.candidate_count,
                        "max_output_tokens": self.max_output_tokens
                        #"stop_sequences": self.stop_sequences
                    }
                )
    
                self.prompt_tokens += response._raw_response.usage_metadata.prompt_token_count
                self.completion_tokens += response._raw_response.usage_metadata.candidates_token_count
                prompt_tokens_k = float(self.prompt_tokens) / 1000.0
                completion_tokens_k = float(self.completion_tokens) / 1000.0
                self.cost = (
                    self.prompt_token_cost * prompt_tokens_k
                    + self.response_token_cost * completion_tokens_k
                )
                self.logger.info(
                    f"This is the response from gemini: {response}"
                    f"\nThis is the cost of the response: {self.cost}"
                )

                if response.candidates[0].finish_reason.value in [1, 2]: # 1 is STOP normal finish_reason and 2 is MAX_TOKENS finish_reason.
                    break  # If finish_reason is either STOP or MAX_TOKENS, exit the loop


        except Exception as e:
            # if "on tokens_usage_based per min" in e.message:
            self.logger.warning(f"Error in gemini: {e}")
            # In case is a Token per Minute Limit error, wait for 61s and call the model again.
            time.sleep(61)
            # response = gemini_model.generate_content(
            #     contents=messages,
            #     generation_config= {
            #         "temperature": self.temperature,
            #         "top_k": self.top_k,
            #         "top_p": self.top_p,
            #         "candidate_count": self.candidate_count,
            #         "max_output_tokens": self.max_output_tokens
            #         #"stop_sequences": self.stop_sequences
            #     }
            # )
            # self.prompt_tokens += response._raw_response.usage_metadata.prompt_token_count
            # self.completion_tokens += response._raw_response.usage_metadata.candidates_token_count
            # prompt_tokens_k = float(self.prompt_tokens) / 1000.0
            # completion_tokens_k = float(self.completion_tokens) / 1000.0
            # self.cost = (
            #     self.prompt_token_cost * prompt_tokens_k
            #     + self.response_token_cost * completion_tokens_k
            # )
            # self.logger.info(
            #     f"There was a problem with gemini: {e}. A break of a min was imposed. After that min, the model has been called again. This is the response from the model: {response}"
            #     f"\nThis is the cost of the response: {self.cost}"
            # )

            return self.chat(messages)
        
        return response

        #     else:
        #         # For other errors, raise the exception again
        #         self.logger.warning("This is not an error related to Token Limits. Check the error!")

        #         raise e    
            

    def get_response_texts(
        self, query_response: Union[List[GenerationResponse], GenerationResponse]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the Gemini model.
        :type query_response: Union[List[GenerationResponse], GenerationResponse]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            response.text
            for response in query_response
        ]