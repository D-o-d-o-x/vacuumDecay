from vacuumDecay.runtime import NeuralRuntime, Runtime, Trainer
from vacuumDecay.utils import choose

def humanVsAi(StateClass, train=True, remember=False, depth=3, bots=[0, 1], noBg=False, start_visualizer=False):
    init = StateClass()
    run = NeuralRuntime(init, start_visualizer=start_visualizer)
    run.game(bots, depth, bg=not noBg)

    if remember or train:
        trainer = Trainer(init)
    if remember:
        trainer.saveToMemoryBank(run.head)
        print('[!] Your cognitive and strategic distinctiveness was added to my own! (Game inserted into memoryBank)')
    if train:
        print("[!] Your knowledge will be assimilated!!! Please stand by.... (Updating Neuristic)")
        trainer.trainFromTerm(run.head)
    print('[!] I have become smart! Destroyer of human Ultimate-TicTacToe players! (Neuristic update completed)')
    print('[!] This marks the beginning of the end of humankind!')
    print('[i] Thanks for playing! Goodbye...')

def aiVsAiLoop(StateClass, start_visualizer=False):
    init = StateClass()
    trainer = Trainer(init, start_visualizer=start_visualizer)
    trainer.train()

def humanVsNaive(StateClass, start_visualizer=False):
    run = Runtime(StateClass(), start_visualizer=start_visualizer)
    run.game()

def main(StateClass):
    options = ['Play Against AI',
               'Play Against AI (AI begins)', 'Play Against AI (Fast Play)', 'Playground', 'Let AI train', 'Play against Naive']
    opt = choose('?', options)
    if opt == options[0]:
        humanVsAi(StateClass)
    elif opt == options[1]:
        humanVsAi(StateClass, bots=[1, 0])
    elif opt == options[2]:
        humanVsAi(StateClass, depth=2, noBg=True)
    elif opt == options[3]:
        humanVsAi(StateClass, bots=[None, None])
    elif opt == options[4]:
        aiVsAiLoop(StateClass)
    elif opt == options[5]:
        humanVsNaive(StateClass)
    else:
        aiVsAiLoop(StateClass)
