class AmbiStoryParser:
    """
    Parse AmbiStory dataset
    """

    def __init__(self, data):
        self.data = data

    def get_samples(self):
        samples = []

        for item_id, item in self.data.items():

            samples.append({
                "id": item_id,
                "homonym": item["homonym"],
                "judged_meaning": item["judged_meaning"],
                "precontext": item["precontext"],
                "sentence": item["sentence"],
                "ending": item.get("ending", ""),
                "example_sentence": item["example_sentence"],
                "full_context": self._build_full_context(item),
                "choices": item["choices"],
                "average": item["average"],
                "stdev": item["stdev"],
                "nonsensical": item["nonsensical"]
            })

        return samples

    def _build_full_context(self, item):

        context = item["precontext"] + " " + item["sentence"]

        if item.get("ending"):
            context += " " + item["ending"]

        return context.strip()