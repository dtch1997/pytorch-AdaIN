# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:11:46 2021
"""

import torch
import torch.nn as nn
import net
import argparse

from function import adaptive_instance_normalization as adain
from torchsummary import summary

class EvalNet(net.Net):
    """ 
    The same as net.Net but without the loss calculation
    Replicate functionality of test.style_transfer()
    Allow saving of model files to single file
    """ 
    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
        g_t = self.decoder(t)
        return g_t
    
def parse_args():
    parser = argparse.ArgumentParser("Script to convert pretrained model to ONNX")
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')
    parser.add_argument('--combined', type=str, default='models/combined.pkl')
    return parser.parse_args()

def load_model(args):
    decoder = net.decoder
    vgg = net.vgg
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    return EvalNet(vgg, decoder)
    
def convert():
    """ Mostly copy pasted from test.py """
    args = parse_args()
    model = load_model(args)
    model.eval()
    torch.save(model, args.combined)
    
def test_convert():
    args = parse_args()
    model = torch.load(args.combined)
    summary(model)
    
if __name__ == "__main__":
    convert()
    test_convert()