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
                rf'([Ll]etra |[Aa]lternativa |[Rr]esposta: ||[Rr]esposta [Cc]orreta: |[Rr]esposta [Cc]orreta e |[Oo]pcao )([{choices_text}])'
            ]

        self.regex_patterns = regex_patterns

    def process_resp(self, text):
        text = text.strip()

        for alternativa in self.choices:
            for suffix in ['.', ')', '-', ':']:
                if text.startswith(alternativa + suffix):
                    return alternativa
                if text.startswith(alternativa + " " + suffix):
                    return alternativa

        for regex in self.regex_patterns:
            match = re.search(regex, text)
            if match:
                return match.group(2)
        
        for i, alternativa in enumerate(self.choices_lower):
            if alternativa in text[:len(alternativa)].lower():
                return self.choices[i]
            
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
        
        # Only works if there diferent enough labels
        #for label in norm_label:
        #    if prediction[:len(label)] == label:
        #        return self.labels[norm_label.index(label)]
        
        if prediction == "":
            return self.fallback
        
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