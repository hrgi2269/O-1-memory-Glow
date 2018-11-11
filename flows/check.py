import torch

def check_tensors(tensor_dict, message):
  for k, v in tensor_dict.items():
    # inf check
    if torch.isinf(v).any():
      print(message)
      print('--Found inf in %s' % k)
      raise FloatingPointError
    # nan check
    if torch.isnan(v).any():
      print(message)
      print('--Found nan in %s' % k)
      raise FloatingPointError
