from sklearn.preprocessing import StandardScaler, MinMaxScaler

standart_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()


def normalize(data, just_transform=False):
	if just_transform:
		return standart_scaler.transform(data)
	else:
		return standart_scaler.fit_transform(data)


def scale(data, just_transform=False):
	if just_transform:
		return min_max_scaler.transform(data)
	else:
		return min_max_scaler.fit_transform(data)