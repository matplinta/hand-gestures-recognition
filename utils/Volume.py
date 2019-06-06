from subprocess import call

class Volume():
    def __init__(self, number=0):
        self.audio_handle = number

    def increase(self, amount):
        self.unmute()
        call(["pactl", "set-sink-volume", str(self.audio_handle), "+" + str(amount)+"%"])
        print(f'Volume increased by: {amount}')

    def decrease(self, amount):
        self.unmute()
        call(["pactl", "set-sink-volume", str(self.audio_handle), "-" + str(amount)+"%"])
        print(f'Volume decreased by: {amount}')

    def mute(self):
        call(["pactl", "set-sink-mute", str(self.audio_handle), "true"])

    def unmute(self):
        call(["pactl", "set-sink-mute", str(self.audio_handle), "false"])
