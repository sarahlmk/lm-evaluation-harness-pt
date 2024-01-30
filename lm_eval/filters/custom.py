from lm_eval.api.filter import Filter
import re
import textdistance
from unidecode import unidecode

class ChoicesFilter(Filter):
    def __init__(
            self,
            choices = None,
            fallback="[invalid]",
            regex_patterns = None
        ) -> None:
        if choices is None:
            choices = ["A", "B", "C", "D", "E"]
        self.choices = choices
        self.fallback = fallback

        self.choices_lower = [a.lower() for a in self.choices]

        if regex_patterns is None:
            choices_text = "".join(choices)
            regex_patterns = [
                fr"(?:[Ll]etra|[Aa]lternativa|[Rr]esposta|[Rr]esposta [Cc]orreta|[Rr]esposta[Cc]orreta e|[Oo]pcao):? ([{choices_text}])\b",
                fr"\b([{choices_text}]) ?[.):-]",
                fr"\b([{choices_text}])$",
                fr"\b([{choices_text}])\b"
            ]
        else:
            regex_patterns = [re.compile(regex) for regex in regex_patterns]
        self.regex_patterns = regex_patterns

    def process_resp(self, text):
        text = text.strip()

        if text in self.choices:
            return text

        for regex in self.regex_patterns:
            match = re.search(regex, text)
            if match:
                return match.group(1)
            
        return self.fallback

    def apply(self, resps, docs) -> None:
        def filter_set(inst):
            return [self.process_resp(resp) for resp in inst]
        return [filter_set(resp) for resp in resps]

class SimilarLabelFilter(Filter):
    def __init__(
            self,
            labels,
            fallback="[invalid]"
        ) -> None:
        self.labels = labels
        self.fallback = fallback

    def process_resp(self, prediction):
        norm_label = [unidecode(s.strip().lower()) for s in self.labels]
    
        prediction = unidecode(prediction.strip().lower())

        if prediction in norm_label:
            return self.labels[norm_label.index(prediction)]
        
        if prediction == "":
            return self.fallback
        
        # Check if there is only one match
        count_matches = 0
        last_match = self.fallback
        for label in norm_label:
            if label in prediction:
                count_matches += 1
                last_match = label
        if count_matches == 1:
            return self.labels[norm_label.index(last_match)]
        
        get_text_until = ['.', ',', ';', ':', '(', ')', '[', ']', '\n']
        for split_char in get_text_until:
            if split_char in prediction:
                prediction = prediction[:prediction.find(split_char)]

        max_length = max([len(s) for s in norm_label])
        prediction = prediction[:max_length]

        similarities = []
        for label in norm_label:
            similarities.append(textdistance.levenshtein.normalized_similarity(prediction, label))
        if max(similarities) < 0.5:
            prediction = self.fallback
        else:
            prediction = self.labels[similarities.index(max(similarities))]

        return prediction

    def apply(self, resps, docs) -> None:
        def filter_set(inst):
            return [self.process_resp(resp) for resp in inst]
        return [filter_set(resp) for resp in resps]