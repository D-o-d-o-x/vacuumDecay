def choose(txt, options):
    while True:
        print('[*] '+txt)
        for num, opt in enumerate(options):
            print('['+str(num+1)+'] ' + str(opt))
        inp = input('[> ')
        try:
            n = int(inp)
            if n in range(1, len(options)+1):
                return options[n-1]
        except:
            pass
        for opt in options:
            if inp == str(opt):
                return opt
        if len(inp) == 1:
            for opt in options:
                if inp == str(opt)[0]:
                    return opt
        print('[!] Invalid Input.')

