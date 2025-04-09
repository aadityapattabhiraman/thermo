#!/home/akugyo/Programs/Python/ragas/bin/python

import os
import asyncio
import typing as t
from langchain_openai import AzureChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import MetricWithLLM, MultiTurnMetric, SingleTurnMetric
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from ragas.metrics.base import MetricType
from ragas.messages import AIMessage, HumanMessage, ToolMessage, ToolCall
from ragas import SingleTurnSample, MultiTurnSample



class Input(BaseModel):

    user_input: str = Field(description="The user request")
    response: str = Field(description="Response from AI")


class Output(BaseModel):

    satisfactory: bool = Field(
        description="Boolean indicating if the response was satisfactory"
    )


class SatisfactoryPrompt(PydanticPrompt[Input, Output]):

    instruction = """
    You are a useful AI assistant. Your task is to classify whether a response
    given by a llm is satisfactory or not. Your output will be True when it is
    satisfactory, else your output will be False.
    """

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                user_input="Can i have some information related to X",
                response="I dont have any information on that. Output is True as it says it doesnt have the information rather than giving a random answer",
            ),
            Output(
                satisfactory=True,
            ),
        ),
        (
            Input(
                user_input="Can i have some information related to X",
                response="""
                Sure here is the info on X,

                these are the steps for Y,
                The reseulting output for this is False as it is exaplaning
                about Y instead of X.
                """
            ),
            Output(
                satisfactory=False,
            ),
        ),
        (
            Input(
                user_input="Can i have some information related to X",
                response="Sure here is the info on X. Output is True as it responds correctly to user query",
            ),
            Output(
                satisfactory=True,
            ),
        ),
    ]



@dataclass
class SatisfactoryRate(MetricWithLLM, MultiTurnMetric, SingleTurnMetric):


    name: str = "satisfactory_rate"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "user_input"}}
    )
    satisfactory_prompt: PydanticPrompt = SatisfactoryPrompt()


    async def _ascore(self, row):

        pass


    async def _single_turn_ascore(self, sample, callbacks):

        prompt_input = Input(
            user_input=sample.user_input, response=sample.response
        )
        prompt_response = await self.satisfactory_prompt.generate(
            data=prompt_input, llm=self.llm
        )

        return int(prompt_response.satisfactory)


    async def _multi_turn_ascore(self, sample, callbacks):

        conversations = sample.user_input
        conversations = [
            message
            for message in conversations
            if isinstance(message, AIMessage) or isinstance(message, HumanMessage)
        ]

        grouped_messages = []
        for msg in conversations:

            if isinstance(msg, HumanMessage):
                human_msg = msg

            elif isinstance(msg, AIMessage) and human_msg:
                grouped_messages.append((human_msg, msg))
                human_msg = None

        grouped_messages = [item for item in grouped_messages if item[0]]
        scores = []

        for turn in grouped_messages:

            prompt_input = Input(
                user_input=turn[0].content,
                response=turn[1].content,
            )
            prompt_response = await self.satisfactory_prompt.generate(
                data=prompt_input,
                llm = self.llm,
            )

            scores.append(prompt_response.satisfactory)

        return sum(scores)


async def trial(user_input, response):

    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    model = LangchainLLMWrapper(model)

    scorer = SatisfactoryRate(llm=model)

    print(user_input)
    print(response)
    sample = SingleTurnSample(user_input=user_input, response=response)
    var = await scorer.single_turn_ascore(sample)
    print("Response:", var)



if __name__ == "__main__":

    user_input = "User: Can you give me troubleshooting steps for TSX505 error 0x0001?"
    response = """
    Assistant: To resolve the 0x0001 error on the TSX505 model, you can follow these troubleshooting steps:

    1. **Check the power source**:
    - Make sure the refrigerator is connected to a working power outlet and that the voltage is correct for the specific model (for example, 120V/60Hz or 230V/50Hz) [doc1][doc2].

    2. **Check the power cord**:
    - Inspect the power cord for any damage or wear. If the cord is damaged, replace it with a new one of the correct type (for example, NEMA 5-15P, EU SCHUKO, UK TYPE G-10A, etc.) [doc1].

    3. **Update the firmware**:
    - If the problem persists, you may need to update the refrigerator's firmware. Download the update zip file from Knowledge Base article #341817 and follow the instructions to update the firmware to version v1.1.03, which resolves issues related to error codes 0x0001 and 0x0002 [doc3].

    4. **Check the internal components**:
    - If the firmware update does not solve the problem, there may be an issue with the internal components. Check the internal power system and connectors to ensure there are no loose connections or damaged components [doc1][doc2].

    5. **Contact technical support**:
    - If none of the above steps solve the problem, contact your local technical support for further assistance. A more thorough maintenance intervention or the replacement of specific components may be necessary [doc3].

    Make sure to follow each step carefully and, if necessary, refer to the service manual for further details.

    To replace the second stage compressor of an ultrafreezer TSX50086V, follow these steps:

    1. **Preparation**:
    - Remove all items from the unit and fully defrost the freezer.
    - Ensure everything is cleaned up in terms of biological safety [doc2].

    2. **Door removal**:
    - Remove the door to prevent any objects from getting stuck inside the unit [doc2].

    3. **Refrigerant and compressor removal**:
    - Have a certified technician remove the refrigerant and the compressor.
    - Discharge the compressor and the system's oil [doc2].

    4. **Disposal of components**:
    - Dispose of the components in accordance with local regulations [doc2].

    5. **Installation of the new compressor**:
    - Install the new compressor following the manufacturer's specific instructions for the TSX50086V model.
    - Ensure all connections are secure and that the system is leak-free.

    6. **System recharge**:
    - Recharge the system with the appropriate refrigerant.
    - Verify that the system is working correctly and that there are no leaks.

    7. **Testing and verification**:
    - Perform a full system test to ensure the new compressor is working correctly.
    - Check the temperature readings on the cabinet sensors and on the compressor bench to ensure they are within the correct parameters [doc3].

    8. **Resetting settings**:
    - Reset the timer when replacing the battery, if necessary.
    - Frequently check the battery conditions by consulting the Power Systems screen [doc4].

    Make sure to follow each step carefully and, if necessary, refer to the service manual for further details.
    """

    asyncio.run(trial(user_input, response))

    
    user_input = "user: What are the root causes for getting negative peaks with the Corona Veo RS detector?"
    response = """
        assistant: The root causes for getting negative peaks with the Corona Veo RS detector and their corresponding remedial actions are as follows:

        1. **Sample solvent and mobile phase differ in composition**:
        - **Remedial Action**: Ensure that the sample is dissolved in the mobile phase [doc1][doc4][doc5].

        2. **Wrong polarization of the analog output interface**:
        - **Remedial Action**: Check the analog output polarization [doc1][doc2][doc3].
        - **Additional Steps**: Apply voltage to the analog out board and verify if the voltage and polarization are correct. Test the functioning of the analog out board as per the instructions in the Installation Guide for the Analog Signal Output Kit [doc2][doc3].

        Please refer to the respective documents for more detailed procedures and additional troubleshooting steps.
    """

    asyncio.run(trial(user_input, response))
