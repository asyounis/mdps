

# Python Imports

# Package Imports

# Ali Package Import

# Project Imports


def get_probability_class(name):

	if(name == "KDE"):
		from kde.kde import KDE
		return KDE

	else:
		print(name)
		assert(False)