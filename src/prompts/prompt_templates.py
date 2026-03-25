class PromptTemplate:

    @staticmethod
    def basic(sample):
        """
        Baseline strategy: Đánh giá đơn giản, hỗ trợ số thập phân.
        """
        return f"""
Rate the plausibility of a word meaning in context.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

Context:
{sample['full_context']}

Scale:
1.0 = Not plausible at all
5.0 = Highly plausible

Instruction: 
Return a single numeric value between 1.0 and 5.0. You are encouraged to use decimals (e.g., 3.5, 4.2) to reflect your confidence levels.
Return ONLY the number.
"""

    @staticmethod
    def criteria(sample):
        """
        Criteria strategy: Phân tích 3 phần của câu chuyện, ép model khắt khe hơn.
        """
        return f"""
You are evaluating whether a proposed meaning fits the narrative based on three parts: the beginning, the sentence itself, and the ending.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

Context Breakdown:
- Beginning: {sample['precontext']}
- Sentence: {sample['sentence']}
- Ending: {sample['ending']}

Evaluation Criteria:
1. How strongly does the beginning suggest this meaning?
2. Does the target sentence support this meaning?
3. Does the ending confirm or contradict this meaning?

Scale:
1.0: Completely implausible/contradictory
5.0: Highly plausible and confirmed

Instruction:
Provide a final plausibility score between 1.0 and 5.0. Decimals (e.g., 2.7, 4.3) are highly preferred to capture subtle nuances. 
Return ONLY the numeric value.
"""

    @staticmethod
    def semeval_official(sample):
        return f"""You are an expert linguist evaluating word meaning plausibility.

Word: {sample['homonym']}
Proposed meaning: {sample['judged_meaning']}

Context:
- Beginning: {sample['precontext']}
- Sentence: {sample['sentence']}
- Ending: {sample['ending']}

Evaluation steps:
1. Does the beginning make the meaning possible?
2. Does the sentence support the meaning?
3. Does the ending confirm or contradict it? (MOST IMPORTANT)

Scoring rules:
- Strong contradiction → 1.0–2.0
- Weak or mixed evidence → 2.5–3.5
- Strong support → 4.0–5.0

Task:
Briefly analyze the context evidence, then provide the final rating between 1.0 and 5.0.
Your final answer must be at the very end in the format: 'Rating: X.X'"""
    @staticmethod
    def improved(sample):
        """
        Rule-based strategy: Dành cho việc tinh chỉnh nhanh dựa trên các quy tắc logic.
        """
        return f"""
Analyze the plausibility of the word meaning within the provided narrative.

Word: {sample['homonym']}
Meaning: {sample['judged_meaning']}

Narrative:
{sample['full_context']}

Scoring Rules:
- Direct contradiction from any part of the context: Score 1.0 - 2.0
- Neutral context or lack of evidence: Score 3.0
- Strong supporting evidence or confirmation: Score 4.0 - 5.0

Instruction:
Return a score between 1.0 and 5.0. Use decimal values (e.g., 3.8) for precise evaluation.
Output format: Just the number.
"""