class Evaluation(object):

    def __init__(self):
        self.episode_cnt = 0
        self.m = dict()

    def update_value(self, m_key, m_value, m_append=None):
        if m_key in self.m:
            self.m[m_key]['value'] += m_value
            self.m[m_key]['cnt'] += 1
        else:
            self.m[m_key] = dict()
            self.m[m_key]['value'] = m_value
            self.m[m_key]['cnt'] = 1

    def summarize(self, key=None):
        if key is None:
            for k in self.m:
                print("Average", k, float(self.m[k]['value'])/self.m[k]['cnt'])

        elif key not in self.m:
            print("Wrong key")

        else:
            print("Average", key, float(self.m[key]['value']) / self.m[key]['cnt'])