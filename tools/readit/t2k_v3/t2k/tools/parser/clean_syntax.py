'''
Pulisce fino alle features. Può anche cancellare le xpos
'''

import sys

sys.stdout.reconfigure(encoding='utf-8')


def main(file):
    formario = open(file, mode="r", encoding='utf-8')
    forme = formario.read()
    formeS = forme.split("\n")
    for forma in formeS:
        if forma != "" and not forma.startswith("#"):
            formaT = forma.split("\t")
            # con xpos
            forma_new = formaT[0] + "\t" + formaT[1] + "\t" + formaT[2] + "\t" + formaT[3] + "\t" + formaT[4] + "\t" + \
                        formaT[5] + "\t_\t_\t_\t" + formaT[9]
            # senza xpos
            # forma_new= formaT[0] + "\t" + formaT[1] + "\t" + formaT[2] + "\t" + formaT[3] + "\t_\t" + formaT[5] + "\t_\t_\t_\t" + formaT[9]
            print(forma_new)
        else:
            print(forma)


main(sys.argv[1])
