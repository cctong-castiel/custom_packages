import numpy as np
import regex as re
import emoji
import logging
import multiprocessing as mp
import time

class AdRemover:

    """It is a class for classifying a post is ad or non-ad"""

    l_emoji = list(map(lambda x: "".join(x.split()), emoji.UNICODE_EMOJI.keys()))

    var = {
        #"slashN": re.compile(r"\n\n"),
        "hashtag": re.compile(r"#"),
        "stock": re.compile(r"\([0-9.A-Z]+\)|（[0-9.A-Z]）"),
        "tel": re.compile(r"(852)*\d{8}\s|(852)*\d{4}-\d{4}\s|(852)*\d{4}\s\d{4}"),
        "emoji": re.compile(r"\b(%s)\b" % "|".join(re.escape(p) for p in l_emoji)),
        "line": re.compile(r"———+|▬▬▬▬+|====+|\+\+\+\+\+\+\+\+|………+|－－－－+|----+"),
        "link": re.compile(r"https?://|bit\.ly/|\w*\.hk/\d+|t.me/"),
        "star": re.compile(r"[*◆◇▷│]"),
        "title": re.compile(r"[【】]"),
        "units": re.compile(r"%|\$|\d+厘|\d+kg|\d+g|\d+折"),
        "punc": re.compile(r"[。！?]"),
        "threshold1": [3.0, 3.0, 2.0, 28.75, 1.0, 2.0, 3.0, 2.0, 3.0],
        "threshold2": [3.0],
        "filter": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }

    def __init__(self, arr):

        self.arr = arr
        self.exist = 3
        self.cutoff = 2

    def regex_item(self, regex):
        return list(map(lambda x: len(regex.findall(str(x))), self.arr))


    def fit_transform(self):

        # multiprocessing
        cpu = mp.cpu_count() - 1
        p = mp.Pool(cpu)

        logging.info("start")
        start1 = time.time()

        l_regex_item2 = np.array(p.map(self.regex_item, [
                                              #self.var["slashN"],
                                              self.var["hashtag"],
                                              self.var["stock"],
                                              self.var["tel"],
                                              self.var["emoji"],
                                              self.var["line"],
                                              self.var["link"],
                                              self.var["star"],
                                              self.var["title"],
                                              self.var["units"],
                                              self.var["punc"]
                                              ]))

        """
        l_regex = [#self.var["slashN"],
                                              self.var["hashtag"],
                                              self.var["stock"],
                                              self.var["tel"],
                                              #self.var["emoji"],
                                              self.var["line"],
                                              self.var["link"],
                                              self.var["star"],
                                              self.var["title"],
                                              self.var["units"],
                                              self.var["punc"]]
        l_regex_item2 = []
        q = mp.Queue()
        procs = [mp.Process(target=self.regex_item, args=(index, q, regex))
                for index, regex in enumerate(l_regex)]
        for proc in procs:
            ret = q.get()
            sorted_ret = [i[i] for i in sorted(ret)]
            l_regex_item2.append(sorted_ret)
        for proc in procs:
            proc.join()"""


        logging.info(f"regex features time span is {time.time() - start1}")

        logging.info(f"the l_regex_item2 is {l_regex_item2} \n size is {l_regex_item2.size}")

        """
        start2 = time.time()
        arr_out1 = np.transpose(
            l_regex_item2[:-1]
        )
        logging.info(f"arr1 time span is {time.time() - start2}")
        start3 = time.time()
        arr_out2 = np.reshape(l_regex_item2[-1], (-1, 1))
        logging.info(f"arr2 time span is {time.time() - start3}")
        """
        """
        start2 = time.time()
        arr_out = np.array(l_regex_item2).T
        logging.info(f"arr_out time span is {time.time() - start2}")"""

        p.close()
        p.join()

        yield l_regex_item2
        #yield self.arr


    """
    def transform(self, arr1, arr2):
        # check if existence more than 3
        arr_binary = np.where(arr1 > 0, 1, 0)
        arr_exist = np.where(arr_binary.sum(axis=1) < self.exist, 0, arr_binary.sum(axis=1))
        # array of thresholds
        arr_up_con = np.where(arr1 > np.array(var["threshold1"]), 1, 0)
        arr_down_con = np.where(arr2 < np.array(var["threshold2"]), 1, 0)
        # horizontal stack
        arr_con = np.hstack((arr_up_con, arr_down_con))
        # calculate weighted value array
        # set cutoff
        arr_con2 = arr_con.sum(axis=1)
        arr_cutoff = arr_exist + arr_con2
        # generate ad cutoff
        arr_ad = np.array([1 if i >= self.cutoff else 0 for i in arr_cutoff], dtype=bool)
        return arr_ad, arr_exist, arr_cutoff
    """