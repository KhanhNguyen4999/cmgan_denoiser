import torch.nn as nn


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return 1 - cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps).mean()

class PearsonCorrelation(nn.Module):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, student_generator_outputs, teacher_generator_outputs):
        
        """
            Loss based on Pearson correlation.
            1 minus Pearson correlation to make it a minimization problem.
            The closer the correlation is to 1, the lower the loss.

            y_s: output of student
            y_t: output of teacher
        """

        loss = pearson_correlation(student_generator_outputs['est_audio'], teacher_generator_outputs['est_audio'])
       
        return loss