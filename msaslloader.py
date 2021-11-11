from torch.utils.data import Dataset, DataLoader
from itertools import repeat
import csv
import os
import torch
from PIL import Image
#import torch.nn.functional as F


FIXED_VID_SIZE = 12


train_sign_list = ["137_afraid", "49_computer", "144_daughter", "56_drink", "176_how_many", "142_can", "123_light", "114_not", "155_late", "189_ready", "127_cheese", "136_buy", "90_work",
"72_day", "79_pencil", "53_write", "174_bread", "140_my", "104_slow", "85_play", "76_hurt", "70_same", "122_teach", "100_apple", "111_have", "99_night", "149_shirt", "92_grandfather", "69_who",
"132_aunt", "81_bad", "82_read", "46_boring", "146_color", "139_jacket", "62_beautiful", "71_nurse", "33_cousin", "75_thank you", "47_please","130_tomorrow", "124_turkey", "115_cook", "78_grandmother",
"105_wrong", "67_name", "117_college", "73_now", "94_right", "110_purple", "91_draw", "51_doctor", "86_sign", "102_shoes", "66_english", "84_dance", "93_woman", "96_pink", "68_you", "95_france", "63_sick",
"80_walk", "34_brother", "58_man", "97_know", "37_nothing", "23_spring", "40_fine", "83_when", "27_sad", "60_understand", "50_help", "24_good", "36_forget", "61_red", "55_different", "57_bathroom", "32_milk",
"74_brown", "44_family", "43_lost", "18_sit", "48_water", "45_hearing", "64_blue", "65_green", "30_where", "38_book", "52_yellow", "35_paper", "28_table", "25_fish", "39_girl", "15_what", "5_happy", "42_boy",
"21_student", "22_learn", "26_again", "4_no", "8_want", "11_sister", "17_friend", "29_must", "0_hello", "13_white", "31_father", "14_chicken", "41_black", "20_yes", "16_tired", "19_mother", "9_deaf", "3_eat",
"6_like", "10_school", "7_orange", "1_nice", "2_teacher", "12_finish"]


class MsaslDataset(Dataset):
	def __init__(self, folder_loc, split_name, transform):
		self.vid_arr, self.sign_class_arr = self.get_video_image_list(folder_loc)
		self.split = split_name
		self.transform = transform
		print(split_name, " split vid length = ", len(self.vid_arr), " sign class length = ", len(self.sign_class_arr))
	def get_video_image_list(self, folder_loc):
		all_vid_arr = []
		all_label_arr = []
		for val in os.listdir(folder_loc):
			for sample in os.listdir(os.path.join(folder_loc, val)):
				vid_content = []
				for img in sorted(os.listdir(os.path.join(folder_loc, val, sample))):
					vid_content.append(os.path.join(folder_loc, val, sample, img))
				all_vid_arr.append(vid_content)
				all_label_arr.append(train_sign_list.index(val))
		return all_vid_arr, all_label_arr
	def __len__(self):
		return len(self.vid_arr)
	def __getitem__(self, idx):
		selected_video_content = self.vid_arr[idx]
		selected_video_sign_class = self.sign_class_arr[idx]
		video_tensor = torch.zeros((FIXED_VID_SIZE, 224, 224, 3))
		cnt = 0
		for img in selected_video_content:
			img_read = Image.open(img).convert('RGB')
			if self.transform:
				img_read = self.transform(img_read)
			img_read = img_read.transpose(0, 1).transpose(1, 2)
			video_tensor[cnt, ...] = img_read
			cnt += 1
		video_tensor = video_tensor.transpose(0, 3).transpose(2, 3).transpose(1, 2)
		label_tensor = torch.tensor(selected_video_sign_class)
		sample = {'video': video_tensor, "video_sign_class": label_tensor}
		return sample

def get_loader(folder_loc, split_name, transform, batch_size, shuffle, num_workers):
	asllvd_dataset = MsaslDataset(folder_loc, split_name, transform)
	data_loader = DataLoader(dataset=asllvd_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	return data_loader