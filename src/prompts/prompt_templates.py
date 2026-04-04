class PromptTemplate:

    @staticmethod
    def chain_of_thought(sample):
        return f"""You are an expert linguist specialized in Lexical Semantics.
Analyze the plausibility of the 'Proposed Meaning' for the 'Target Word' based on a three-part narrative.

[DATA]
Target Word: {sample['homonym']}
Proposed Meaning: {sample['judged_meaning']}
---
Context:
1. Beginning: {sample['precontext']}
2. Sentence: {sample['sentence']}
3. Ending: {sample['ending']}

[TASK]
Follow these steps strictly:
Step 1 (Contextual Flow): Analyze how the 'Beginning' and 'Sentence' set up the target word. Is the meaning possible at this stage?
Step 2 (The Ending Test): Critically evaluate the 'Ending'. Does it provide a twist that contradicts the meaning, or does it confirm it? (Crucial for AmbiStory).
Step 3 (Conflict Detection): Identify any semantic clashes between the 'Proposed Meaning' and the physical/logical world described in the full story.

[SCORING SCALE]
- 1.0 to 1.5: Direct logical contradiction (e.g., the ending says the object is broken, but meaning implies it's working).
- 2.0 to 3.0: High improbable or lacks any supporting evidence.
- 3.5 to 4.5: Highly plausible, fits the narrative flow well.
- 5.0: Perfectly confirmed by all parts of the context.

Final Answer: Briefly state your reasoning for each step, then end with 'Rating: X.X'"""
    
    @staticmethod
    def one_shot(sample):
        return f"""You are an expert linguist. Rate the word meaning plausibility from 1.0 to 5.0.

[EXAMPLE]
Context:
- Beginning: The professional chef was preparing a specialized fish dish.
- Sentence: He reached for the scale to ensure the portions were exact.
- Ending: He then used the knife to remove the skin and bones.
Word: scale
Proposed Meaning: a device used to weigh object.
Analysis: The beginning mentions portions, and the sentence mentions exactness, which supports weighing. The ending doesn't contradict it.
Rating: 4.8

[YOUR TASK]
Context:
- Beginning: {sample['precontext']}
- Sentence: {sample['sentence']}
- Ending: {sample['ending']}
Word: {sample['homonym']}
Proposed Meaning: {sample['judged_meaning']}

Analysis: (Analyze briefly)
Rating: """

    @staticmethod
    def few_shot(sample):
        return f"""You are a linguistic expert evaluating the plausibility of word senses in narratives.
Rate the following on a scale of 1.0 (Implausible) to 5.0 (Perfectly Plausible).

[EXAMPLES]
1. Plausible Case:
Context: [He was hiking up the mountain. / He reached the peak just before sunset. / The view from the top was breathtaking.]
Word: peak | Meaning: the pointed top of a mountain.
Rating: 5.0

2. Contradictory Case:
Context: [She wanted to play music. / She sat down at the organ to practice. / Then she went to the hospital for her kidney transplant.]
Word: organ | Meaning: a large musical instrument with pipes.
Rating: 1.2 (Reason: The ending reveals 'organ' refers to a biological body part, not the instrument).

3. Ambiguous/Neutral Case:
Context: [He went to the bank. / He stood there for a long time. / The weather was getting cold.]
Word: bank | Meaning: a financial institution.
Rating: 3.0 (Reason: Not enough context to confirm if it's a river bank or a money bank).

[CURRENT TASK]
Context:
- Beginning: {sample['precontext']}
- Sentence: {sample['sentence']}
- Ending: {sample['ending']}
Word: {sample['homonym']}
Proposed Meaning: {sample['judged_meaning']}

Brief Analysis:
Rating: """