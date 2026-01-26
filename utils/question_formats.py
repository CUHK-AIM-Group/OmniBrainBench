def get_multiple_choice_prompt(question,choices,is_reasoning = False,lang = "en"):
    choices = [str(choice) for choice in choices]
    options = "\n".join(choices)

    if lang == "en":
        prompt = f"""
Question: {question}
Options: 
{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + (
                "Please provide step-by-step reasoning (explain your chain of thought). "
                "After your reasoning, on a new line write: Final Answer: \"\\boxed{<letter>}\". "
                "Only put the chosen option letter inside the box."
            )
        else:
            prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly." 

    elif lang == "zh":
        prompt = f"""
问题： {question}
选项： 
{options}"""
        if is_reasoning:
            prompt = prompt + "\n" + (
                "请先逐步说明你的推理过程（给出要点和推理步骤）。"
                " 在新的行中写出最终答案：Final Answer: \"\\boxed{<字母>}\"。"
                " 仅将选项字母放在括号内。"
            )
        else:
            prompt = prompt + "\n" +  "请直接使用给定选项中的选项字母来回答该问题。"
    return prompt

def get_judgement_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = (
                question + "\n" +
                "Please provide step-by-step reasoning (brief chain-of-thought). "
                "Then, on a new line, write: Final Answer: \"\\boxed{yes}\" or \"\\boxed{no}\"."
            )
        else:
            prompt = question + "\n" + "Please output 'yes' or 'no'(no extra output)."
    elif lang == "zh":
        if is_reasoning:
            prompt = (
                question + "\n" +
                "请先逐步说明你的推理过程（简要的思路和依据）。"
                " 然后在新的一行写出最终答案：Final Answer: \"\\boxed{是}\" 或 \"\\boxed{否}\"。"
            )
        else:
            prompt = question + "\n" + "请输出'是'或'否'(不要有任何其它输出)。"
    return prompt

def get_close_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = (
                question + "\n" +
                "Please give brief step-by-step reasoning, then on a new line write: Final Answer: \"\\boxed{<word or phrase>}\"."
            )
        else:
            prompt = question + "\n" + "Answer the question using a single word or phrase."
    elif lang == "zh":
        if is_reasoning:
            prompt = (
                question + "\n" +
                "请先给出简要的逐步推理，然后在新的一行写出最终答案：Final Answer: \"\\boxed{<单词或短语>}\"。"
            )
        else:
            prompt = question + "\n" + "请用一个单词或者短语回答该问题。"
    return prompt

def get_open_ended_prompt(question,is_reasoning = False, lang = "en"):
    if lang == "en":
        if is_reasoning:
            prompt = (
                question + "\n" +
                "Please provide step-by-step reasoning (concise chain-of-thought). "
                "Then on a new line give a concise final answer in the format: Final Answer: \"\\boxed{<answer>}\"."
            )
        else:
            prompt = question + "\n" + "Please answer the question concisely."
    elif lang == "zh":
        if is_reasoning:
            prompt = (
                question + "\n" +
                "请先给出简要的逐步推理（要点式说明），然后在新的一行给出简洁的最终答案，格式为：Final Answer: \"\\boxed{<答案>}\"。"
            )
        else:
            prompt = question + "\n" + "请简要回答该问题。"
    return prompt

def get_report_generation_prompt():
    prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."
    return prompt



