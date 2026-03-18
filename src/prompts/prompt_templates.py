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
        return f"""
    As an expert linguist, evaluate the plausibility of a word's meaning in a story on a scale of 1.0 to 5.0. 
    Human annotators often give middle-range scores (2.0 - 4.0) when the context is ambiguous or could support multiple meanings.

    ### EXAMPLES FOR CALIBRATION:

    Example 1:
    - WORD: track
    - MEANING: a pair of parallel rails providing a runway for wheels
    - CONTEXT: Beginning: train station... Sentence: They followed the track. Ending: run along the railway line, hopping from sleeper to sleeper.
    - ANALYSIS: The ending explicitly mentions "railway line" and "sleeper", strongly confirming the physical rail meaning.
    - SCORE: 3.6 (High, but not 5.0 because "track" could also metaphorically mean the trail they found).

    Example 2:
    - WORD: track
    - MEANING: evidence pointing to a possible solution
    - CONTEXT: Beginning: train station... Sentence: They followed the track. Ending: found interesting clues that helped them solve the problem.
    - ANALYSIS: The ending focuses on "clues" and "solving the problem", making the "evidence" meaning more plausible than the physical rail.
    - SCORE: 4.2

    Example 3:
    - WORD: track
    - MEANING: a pair of parallel rails providing a runway for wheels
    - CONTEXT: Beginning: train station... Sentence: They followed the track. Ending: (None/Empty)
    - ANALYSIS: Without an ending, the word is highly ambiguous. It could be the rail or the evidence.
    - SCORE: 3.0

    ### CURRENT TASK:
    - WORD: {sample['homonym']}
    - PROPOSED MEANING: {sample['judged_meaning']}

    STORY:
    1. Beginning: {sample['precontext']}
    2. Target Sentence: {sample['sentence']}
    3. Ending: {sample['ending']}

    EVALUATION PROTOCOL:
    - Phase 1: Assess if the 'Beginning' makes the 'Meaning' possible.
    - Phase 2: Assess if the 'Target Sentence' uses the word in a way that fits the 'Meaning'.
    - Phase 3: Does the 'Ending' confirm, contradict, or leave the meaning ambiguous?
    - Phase 4: Match human tendency. If the story allows for another interpretation, stay within the 2.5 - 3.8 range.

    OUTPUT FORMAT: Return ONLY a single decimal number (e.g., 3.2). No explanation.
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