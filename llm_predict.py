import openai

file_path = 'llmoutput\prediction.txt'


def openai_reply(content, apikey):
    openai.api_key = apikey
    response = openai.ChatCompletion.create(
        model="YOUR MODEL",
        messages=[
            {"role": "user", "content": content}
        ],
        temperature=0.5,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    content_list = []
    head1 = 'The prerequisite relationship is defined as follows: if knowledge concept A is a prerequisite concept for knowledge concept B, then knowledge point A needs to be mastered before knowledge concept B can be learned, because the understanding of knowledge concept B depends on knowledge concept A. For example, “address mapping” is a prerequisite concept for “virtual address spaces”; “operands” is a prerequisite concept for “assignment statements”; “priority control” is a prerequisite concept for “multiprocessor scheduling”.\n'
    head2 = 'You will serve as a pedagogical specialist in the field of computer science to help me predict whether there is a prerequisite relationship between two given knowledge concepts. \n'
    head3 = ' I will provide you with some knowledge concept-pairs in the form of lists, e.g., [knowledge concept A, knowledge concept B]. Please provide your prediction result in the following list format: [“knowledge concept A”, “knowledge concept B”, “prediction result” (If there exists a prerequisite relationship, please set the prediction result to "1"; if there does not exist a prerequisite relationship, please set the prediction result to "0"; if you are unable to make a prediction, please set the prediction result to " None"), “reasoning” (brief explain your reasoning for the prediction)].\n '
    head4 = 'Requirements:\n 1. Please ensure that the “reasoning ” is no longer than 100 words.\n 2. Do not provide any other text outside the list string.'
    head = head1 + head2 + head3 + head4
    with open("./concept_pairs.txt", 'r', encoding="utf-8") as f:
        for line in f:
            line = line.replace('\n', '')
            content_list.append(line)

    content = ''
    res = []
    for i, line in enumerate(content_list):
        content += str(line) + '\n'
        ans = ''
        if (i + 1) % 30 == 0 or (i + 1) == len(content_list):
            head_conten = head + content
            ans = openai_reply(head_conten, 'YOUR API KEY')
            ans = str(ans).replace('\n\n', '\n')
            content = ''
            with open(file_url1, "a", encoding="utf-8") as f:
                f.write(ans)
                f.write('\n')
