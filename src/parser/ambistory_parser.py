class AmbiStoryParser:
    """
    Parse AmbiStory dataset (robust version)
    """

    def __init__(self, data):
        self.data = data

    def clean(self, text):
        return " ".join(text.split()) if isinstance(text, str) else ""

    def get_samples(self):
        samples = []

        for item_id, item in self.data.items():

            avg = item.get("average", None)
            if isinstance(avg, str) and not avg.replace('.', '').isdigit():
                avg = None

            sample = {
                "id": item_id,
                "homonym": item.get("homonym", ""),
                "judged_meaning": item.get("judged_meaning", ""),
                "precontext": self.clean(item.get("precontext", "")),
                "sentence": self.clean(item.get("sentence", "")),
                "ending": self.clean(item.get("ending", "")),
                "example_sentence": item.get("example_sentence", ""),
                "full_context": self._build_full_context(item),
                "choices": item.get("choices", []),
                "average": avg,
                "stdev": item.get("stdev", None),
                "nonsensical": item.get("nonsensical", None),
            }

            samples.append(sample)

        return samples

    def _build_full_context(self, item):
        pre = item.get("precontext", "")
        sent = item.get("sentence", "")
        end = item.get("ending", "")

        context = f"{pre} {sent}"
        if end:
            context += f" {end}"

        return " ".join(context.split()).strip()