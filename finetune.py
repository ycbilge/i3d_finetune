import pickle
import torch
import torch.optim as optim
from i3d_model import I3D, Unit3Dpy
from torchvision import transforms
from msaslloader import get_loader
from torch.autograd import Variable
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm

train_loc = '/home/yunus/Documents/ProtoNetVersions/I3D_finetune/finetune/finetune_msasl/train/'
test_loc = '/home/yunus/Documents/ProtoNetVersions/I3D_finetune/finetune/finetune_msasl/test/'

batch_size = 24

shuffle_bool_train = True
shuffle_bool_test = False
num_workers = 0
learning_rate = 0.001
momentum = 0.5
n_epoch = 100


criterion = nn.CrossEntropyLoss() #softmax yapılıyor

def save_model(model):
	path = "./finetuned_model/i3d_finetuned.pth"
	torch.save(model.state_dict(), path)


def get_i3d_model():
	i3d_rgb = I3D(num_classes=400, modality='rgb')
	i3d_rgb.load_state_dict(torch.load('/home/yunus/Documents/ProtoNetVersions/I3D/extract_features/model/model_rgb.pth'))
	i3d_rgb.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024, out_channels=120, kernel_size=(1, 1, 1), activation=None, use_bias=True, use_bn=False)
	i3d_rgb.cuda()
	return i3d_rgb

def evaluate(i3d_rgb, test_dl):
	accuracy = 0.0
	total = 0.0
	i3d_rgb.eval()
	with torch.no_grad():
		for i, sample in enumerate(test_dl):
			vid_tensor = sample['video'].cuda()
			labels = sample['video_sign_class']#.cuda()
			outputs, _ = i3d_rgb(vid_tensor)
			_, predicted = torch.max(outputs.data.cpu(), 1)
			total += labels.size(0)
			accuracy += (predicted == labels).sum().item()
	accuracy = (100 * accuracy / total)
	return accuracy


def train_and_test(i3d_rgb, optimizer, train_dl, test_dl):
	best_accuracy = -9999.0
	for epoch in range(n_epoch):
		i3d_rgb.train()
		running_loss = 0
		pbar = tqdm(total=len(train_dl))
		total = 0
		for i, sample in enumerate(train_dl):
			vid_tensor = sample['video'].cuda()
			labels = sample['video_sign_class'].cuda()
			optimizer.zero_grad()
			outputs, _ = i3d_rgb(vid_tensor)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss  += loss.item()
			pbar.update()
			total += labels.data.cpu().size(0)
		accuracy = evaluate(i3d_rgb, test_dl)
		if accuracy >= best_accuracy:
			best_accuracy = accuracy
			print("Model is saved and updated.") 
			save_model(i3d_rgb)
		print("Total label size = ", total)
		print("Epoch = ", epoch+1, " Loss = ", running_loss/3160.0, " Accuracy = ", accuracy)

def main():
	transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
	i3d_rgb = get_i3d_model()
	optimizer = optim.Adam(i3d_rgb.parameters(), lr=learning_rate)#0.001, momentum=0.9)
	train_dl = get_loader(train_loc, 'train', transform, batch_size, shuffle_bool_train, num_workers)
	test_dl = get_loader(test_loc, 'test', transform, batch_size, shuffle_bool_test, num_workers)
	train_and_test(i3d_rgb, optimizer, train_dl, test_dl)
	



if __name__ == "__main__":
	main()