#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python3 configure.py ..." has to be run

import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser(
    description="Creates anima scripts config file for using scripts.")
parser.add_argument('-s', '--scripts-public', type=str, default="~/Anima-Scripts-Public/", help="Anima scripts public folder")
parser.add_argument('-S', '--scripts-private', type=str, default="~/Anima-Scripts/", help="Anima scripts private folder")
parser.add_argument('-d', '--scripts-data', type=str, default="~/Anima-Scripts-Data-Public/", help="Anima scripts data folder")
parser.add_argument('-a', '--anima', type=str, default="~/Anima-Public/build/bin/", help="Anima executables folder")

args = parser.parse_args()

homeFolder = os.path.expanduser('~')
configFolder = os.path.join(homeFolder, ".anima")
if not os.path.exists(configFolder):
    os.mkdir(configFolder)

configFilePath = os.path.join(configFolder, "config.txt")
configFile = open(configFilePath, "w")

configFile.write("# Variable names and section titles should stay the same\n")
configFile.write("# Make the anima variable point to your Anima public build\n")
configFile.write("# Make the extra-data-root point to the data folder of Anima-Scripts\n")
configFile.write("# The last folder separator for each path is crucial, do not forget them\n")
configFile.write("# Use full paths, nothing relative or using tildes\n\n")

configFile.write("[anima-scripts]\n")

animaScriptsPublicPath = os.path.abspath(os.path.expanduser(os.path.normpath(args.scripts_public))) + os.sep
configFile.write("anima-scripts-public-root = " + animaScriptsPublicPath + "\n")

animaScriptsPrivatePath = os.path.abspath(os.path.expanduser(os.path.normpath(args.scripts_private))) + os.sep
configFile.write("anima-scripts-root = " + animaScriptsPrivatePath + "\n")

animaPath = os.path.abspath(os.path.expanduser(os.path.normpath(args.anima))) + os.sep
configFile.write("anima = " + animaPath + "\n")

dataPath = os.path.abspath(os.path.expanduser(os.path.normpath(args.scripts_data))) + os.sep
configFile.write("extra-data-root = " + dataPath + "\n")

configFile.close()
