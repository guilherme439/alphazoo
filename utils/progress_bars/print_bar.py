
class PrintBar():

    def __init__(self, title, bar_steps, bar_divisions):
        self.bar_divisions = bar_divisions
        self.bar_steps = bar_steps

        self.current_progress = 0
        self.interval_size = bar_steps / bar_divisions
        self.next_step = self.interval_size

        print(title + "  |", end='', flush=True)
        

    def next(self):
        while self.current_progress+1 >= self.next_step:
            print('■', end=' ', flush=True)
            self.next_step += self.interval_size
        self.current_progress +=1
    
    def finish(self):
        print('|', end='\n', flush=True)