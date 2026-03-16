class PromptTemplate:

    @staticmethod
    def basic(sample):

        return f"""
Rate the plausibility of a word meaning in context.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

Context:
{sample['full_context']}

Scale:
1 = Not plausible
5 = Highly plausible

Return only a number between 1 and 5.
"""

    @staticmethod
    def criteria(sample):

        return f"""
You are evaluating whether a proposed meaning fits the narrative.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

Beginning:
{sample['precontext']}

Sentence:
{sample['sentence']}

Ending:
{sample['ending']}

Evaluate:

1. Does the beginning suggest this meaning?
2. Does the sentence support it?
3. Does the ending confirm it?

Scale:
1 completely implausible
5 highly plausible

Return only an integer from 1 to 5.
"""

    @staticmethod
    def improved(sample):

        return f"""
Evaluate plausibility of a word meaning.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

Narrative:
{sample['full_context']}

Rules:
- If ending contradicts meaning → score ≤2
- If evidence unclear → score 3
- If strongly confirmed → score 5

Return only one number 1-5.
"""