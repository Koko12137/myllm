import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from models.modeling import MyLLMForCausalLM
from models.configuration import MyLLMConfig


class MyLLMDistillWrapper(nn.Module):
    student: MyLLMForCausalLM
    teacher: AutoModelForCausalLM
    min_kl: float
    max_kl: float
    loss_kl: nn.KLDivLoss
    
    def __init__(self, teacher: str, config: MyLLMConfig, min_kl: float = 0.3, max_kl: float = 0.7) -> None:
        super().__init__()
        self.config = config
        self.min_kl = min_kl
        self.max_kl = max_kl
        
        # Load the teacher model
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher, 
            torch_dtype=torch.float16, 
        )
        # Freeze the teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        # Switch the teacher model to eval mode
        self.teacher.eval()
        
        # Initialize the loss function
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')
        
    def student_from_pretrained(self, pretrained: str, debug: bool = False) -> None:
        # Load the student model from pretrained
        self.student = MyLLMForCausalLM.from_pretrained(pretrained, debug=debug)
        
    def new_student(self, config: MyLLMConfig, debug: bool = False) -> None:
        # Create a new student model
        self.student = MyLLMForCausalLM(config, debug=debug)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None, 
        inputs_embeds: torch.FloatTensor = None, 
        position_ids: torch.LongTensor = None, 
        attention_mask: torch.Tensor = None, 
        # Training arguments
        labels: torch.LongTensor = None, 
        # Output arguments
        use_cache: bool = None,                     
        output_attentions: bool = None, 
        output_hidden_states: bool = None, 
        return_dict: bool = None, 
        num_logits_to_keep: int = 0, 
        # Cache arguments
        past_key_values: Cache = None,              
        cache_position: torch.LongTensor = None,    
        # Training step
        step: int = 0,  # Current training step
        total_steps: int = 10000,  # Total training steps
    ) -> tuple | CausalLMOutputWithPast:
        # Forward pass for the student model
        student_outputs: BaseModelOutputWithPast = self.student(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds, 
            position_ids=position_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=True, 
            num_logits_to_keep=num_logits_to_keep, 
            past_key_values=past_key_values, 
            cache_position=cache_position, 
        )
        
        # Get the loss from the student model
        if labels is not None:
            loss = student_outputs.loss
        
        if self.training:
            # Teacher forward pass with no grad
            with torch.no_grad():
                teacher_outputs: BaseModelOutputWithPast = self.teacher(
                    input_ids=input_ids, 
                    inputs_embeds=inputs_embeds, 
                    position_ids=position_ids, 
                    attention_mask=attention_mask, 
                    labels=None, 
                    use_cache=use_cache, 
                    output_attentions=output_attentions, 
                    output_hidden_states=output_hidden_states, 
                    return_dict=True, 
                    num_logits_to_keep=num_logits_to_keep, 
                    past_key_values=past_key_values, 
                    cache_position=cache_position, 
                )
                
            # Only compute KL divergence if labels are provided
            if labels is not None:
                # Compute KL weight (e.g., linearly decay and stop at a minimum value)
                kl_weight = max(self.min_kl, self.max_kl - step / total_steps) 
                
                # Calculate the KL divergence loss
                kl_loss = self.loss_kl(
                    nn.functional.softmax(
                        student_outputs.logits.reshape(
                            -1, student_outputs.logits.size(-1)
                        ), 
                        dim=-1
                    ),
                    nn.functional.softmax(
                        teacher_outputs.logits.reshape(
                            -1, teacher_outputs.logits.size(-1)
                        )[:, :student_outputs.logits.size(-1)], 
                        dim=-1, 
                    )
                )
                
                # Scale KL loss by kl_weight
                loss = (1 - kl_weight) * loss + kl_weight * kl_loss

            # Replace the loss in the student outputs
            student_outputs.loss = loss
        
        if return_dict:
            return student_outputs
        else:
            return student_outputs.to_tuple()
