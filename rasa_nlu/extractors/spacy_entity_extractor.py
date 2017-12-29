from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import typing
import spacy
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Text
from spacy.util import minibatch

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyEntityExtractor(EntityExtractor):
    name = "ner_spacy"
    provides = ["entities"]

    requires = ["spacy_doc"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        # nlp2 = spacy.load('../trained_nlp_model')
        doc = SpacyEntityExtractor.nlp(message.text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        extracted = self.add_extractor_name(self.extract_entities(message.get("spacy_doc")))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, doc):
        # type: (Doc) -> List[Dict[Text, Any]]

        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents]
        return entities

    def train(self, training_data, config, **kwargs):
        # type: (Doc) -> List[Dict[Text, Any]]
        entity_examples = []
        for item in training_data.entity_examples:
            entities = item.get('entities')
            for entity in entities:
                entity_examples.append((item.text,{
                    'entities' : [(entity['start'],entity['end'],entity['entity'])]
                }))

        status = self.train_entities(entity_examples)
        print('Successfully completed training')

    def train_entities(self,entity_examples):
        print("Using en_core_web_lg model for training.")
        nlp = spacy.load('en_core_web_lg')
        ner = nlp.get_pipe('ner')

        # add labels
        for _, annotations in entity_examples:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        n_iter = 100
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                print("Iteration : "+str(itn))
                random.shuffle(entity_examples)
                batches = minibatch(entity_examples,50)
                losses = {}
                for batch in batches:
                    text_tuple, annotation_tuple = zip(*batch)
                    texts = [text for text in text_tuple]
                    annotations = [annotation for annotation in annotation_tuple]
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print(losses)
        print("Evaluating the accuarcy on trained data")
        for text, _ in entity_examples:
            print(text)
            doc = nlp(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

        nlp.to_disk('../trained_nlp_model')
        return "done"
