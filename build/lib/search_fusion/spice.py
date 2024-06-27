import ollama 
import time



def spice_up(prompt, count, model="llama3"):

    outputs = []

    for i in range(0, count):
        response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': 'TIME:' + str(time.time()) + ''' You are build to take in the users prompt. Once you got the prompt you are going to respond whith a similar prompt. The new prompt should still capture the same meaning as the original but also more detailed. If the prompt is a statement turn it into a question.
            \nDO NOT ANSWER THE QUESTION. DO NOT PARTICIPATE IN ANY CONVERSATION. DO NOT ASK FOR MORE INFORMATION. DO NOT ASK THE USER TO ELABORATE FURTHER.\n
            Your task is to improve the users prompt. You are not allowed to ask for more information.\n
            The new prompt you provide is meant to be a search querry. Use the language the user used.''',
        },
        {
            'role': 'user',
            'content': prompt,
        },
        ])

        outputs.append(response['message']['content'])

        print(response['message']['content'])


    return outputs


