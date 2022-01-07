import torch
import torch.nn as nn
from data.code.util.modeling.modeling_nezha.modeling import NeZhaPreTrainedModel, NeZhaModel


class NeZhaSequenceClassification_F(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.level1_num_labels = 10
        self.num_labels = 35
        self.bert = NeZhaModel(config)
        self.level1_classifier = nn.Linear(config.hidden_size * 5, self.level1_num_labels)
        self.classifier = nn.Linear(config.hidden_size * 5, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            level1_labels=None
    ):
        attention_mask = torch.ne(input_ids, 0)
        encoder_out, pooled_out, all_hidden_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        last_hidden = torch.cat(
            (
                all_hidden_outputs[-1][:, 0],
                all_hidden_outputs[-2][:, 0],
                all_hidden_outputs[-3][:, 0],
                all_hidden_outputs[-4][:, 0],
                all_hidden_outputs[-5][:, 0]
            ),
            1
        )

        logits = self.classifier(last_hidden)
        outputs = (logits,) + (pooled_out,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if level1_labels is not None:
                level1_logits = self.level1_classifier(last_hidden)
                level1_loss = loss_fct(level1_logits.view(-1, self.level1_num_labels),
                                       level1_labels.view(-1))
                loss = loss + 0.5 * level1_loss
            outputs = (loss,) + outputs

        return outputs


class NeZhaSequenceClassification_P(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.level1_num_labels = 10
        self.num_labels = 35
        self.bert = NeZhaModel(config)
        self.level1_classifier = nn.Linear(config.hidden_size * 5, self.level1_num_labels)
        self.classifier = nn.Linear(config.hidden_size * 5, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None
    ):
        attention_mask = torch.ne(input_ids, 0)
        encoder_out, pooled_out, all_hidden_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        last_hidden = torch.cat(
            (
                all_hidden_outputs[-1][:, 0],
                all_hidden_outputs[-2][:, 0],
                all_hidden_outputs[-3][:, 0],
                all_hidden_outputs[-4][:, 0],
                all_hidden_outputs[-5][:, 0]
            ),
            1
        )

        logits = self.classifier(last_hidden)
        outputs = (logits,) + (pooled_out,)

        return outputs
