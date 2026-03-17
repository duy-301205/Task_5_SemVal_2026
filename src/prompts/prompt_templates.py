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
        """
        Chế độ chuẩn NCKH: Sử dụng Calibration (Hiệu chuẩn) để khớp với nhãn trung bình của người chấm.
        Sử dụng Persona 'Linguist' để tăng độ sâu suy luận.
        """
        return f"""
As an expert linguist specializing in narrative ambiguity, your task is to evaluate the plausibility of a specific word meaning within a three-part story (Beginning, Target Sentence, Ending).

WORD: {sample['homonym']}
PROPOSED MEANING: {sample['judged_meaning']}

STORY PROGRESSION:
1. Beginning: {sample['precontext']}
2. Target Sentence: {sample['sentence']}
3. Ending: {sample['ending']}

EVALUATION PROTOCOL:
- Phase 1: Does the 'Beginning' set a context that makes the 'Proposed Meaning' likely?
- Phase 2: Does the 'Target Sentence' reinforce this meaning, or could it refer to a different homonym?
- Phase 3 (Critical): Does the 'Ending' confirm or provide a 'twist' that contradicts the 'Proposed Meaning'?

SCORING CALIBRATION (Strict Scale):
- 5.0 (Perfect Fit): All three parts of the story clearly and exclusively support this meaning.
- 4.0 (High Plausibility): The meaning fits well, but the context is slightly generic.
- 3.0 (Ambiguous): The context is neutral; it could be this meaning or another. There is no strong evidence either way.
- 2.0 (Low Plausibility): There is a slight mismatch or the ending subtly suggests a different interpretation.
- 1.0 (Impossible/Contradiction): The Ending or context explicitly reveals that the word refers to something else.

IMPORTANT: Humans gave an average score here. Do not just pick 1.0 or 5.0 unless it is absolute. If the story is even slightly tricky or vague, the score SHOULD be between 2.5 and 4.0. Be a critical and skeptical judge.

OUTPUT FORMAT: Return ONLY a single decimal number (e.g., 2.4, 3.7, 4.8). No explanation.
"""


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