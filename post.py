import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--token", help="token of dir", default="", type=str)
parser.add_argument("--mpi", action='store_true')
args = parser.parse_args()


os.system("./clean.sh")

# if(args.mpi):
path="bghosh@contact.mpi-sws.org:/home/bghosh/Desktop/pac_explanation/"
os.system("tar -czvf file_to_send.tar.gz pac_explanation/* data/model/* data/objects/* data/raw/* *txt *md *.py *sh *.ipynb")
os.system("rsync -vaP file_to_send.tar.gz "+path)


# else:
#     path="nscc:/home/projects/11000744/bishwa/xRNN" +args.token+"/" 
#     os.system("tar -czvf file_to_send.tar.gz ltlf2dfa/* RNN2DFA/* PACTeacher/* samples2ltl/* *.py *.sh *.ipynb *.pbs")
#     os.system("rsync -vaP file_to_send.tar.gz "+path)


os.system("rm file_to_send.tar.gz")