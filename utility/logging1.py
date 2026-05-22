import os
from datetime import datetime

class Logger():
    def __init__(self, filename, is_debug, path='./logs/'):
        self.filename = filename
        self.path = path
        self.log_ = not is_debug

    def logging(self, s):
        s = str(s)


        print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), s)

        if self.log_:

            safe_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


            safe_filename = self.filename.replace(':', '-').replace(' ', '_')


            file_path = os.path.join(self.path, f"{safe_timestamp}_{safe_filename}")


            os.makedirs(self.path, exist_ok=True)


            with open(file_path, 'a+', encoding='utf-8') as f_log:
                f_log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {s}\n")

