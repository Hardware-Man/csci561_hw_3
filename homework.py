from numpy import exp, array, random, zeros, clip, ndarray, argmax, where
from numpy import sum as npsum
from numpy import max as npmax
from pandas import DataFrame
import matplotlib.pyplot as plt
import csv
import sys
from typing import Dict

# Ranked by median income
zip_code_lookup = {
    '10454': 1, '11212': 2, '10455': 3, '10456': 4, '10460': 5,
    '10035': 6, '10474': 7, '10452': 8, '10029': 9, '10451': 10,
    '10453': 11, '10459': 12, '10472': 13, '10002': 14, '11224': 15,
    '10457': 16, '10458': 17, '10468': 18, '10467': 19, '10030': 20,
    '11219': 21, '11207': 22, '11355': 23, '10473': 24, '10037': 25,
    '11213': 26, '11206': 27, '11208': 28, '10039': 29, '11354': 30,
    '11233': 31, '11692': 32, '10040': 33, '11691': 34, '10032': 35,
    '10475': 36, '11220': 37, '10031': 38, '11223': 39, '10027': 40,
    '11214': 41, '11235': 42, '11204': 43, '10462': 44, '11005': 45,
    '11230': 46, '10461': 47, '11203': 48, '11373': 49, '11368': 50,
    '11372': 51, '10466': 52, '10463': 53, '10034': 54, '10304': 55,
    '11693': 56, '10026': 57, '11237': 58, '11369': 59, '11226': 60,
    '11432': 61, '11377': 62, '11221': 63, '11370': 64, '11225': 65,
    '11205': 66, '11433': 67, '11229': 68, '11423': 69, '10303': 70,
    '10470': 71, '10469': 72, '11435': 73, '11367': 74, '10033': 75,
    '11106': 76, '11365': 77, '11356': 78, '11236': 79, '10302': 80,
    '11358': 81, '11416': 82, '11421': 83, '10305': 84, '11232': 85,
    '11228': 86, '11385': 87, '11210': 88, '10009': 89, '11418': 90,
    '10301': 91, '11436': 92, '11417': 93, '11374': 94, '10465': 95,
    '11415': 96, '11104': 97, '11218': 98, '11103': 99, '11427': 100,
    '11209': 101, '11419': 102, '11216': 103, '11378': 104, '11414': 105,
    '11234': 106, '11357': 107, '11694': 108, '11379': 109, '10306': 110,
    '11420': 111, '11211': 112, '11429': 113, '11364': 114, '11360': 115,
    '11105': 116, '11412': 117, '11361': 118, '10314': 119, '11102': 120,
    '11422': 121, '10310': 122, '11375': 123, '10471': 124, '11363': 125,
    '10036': 126, '11101': 127, '10038': 128, '11004': 129, '11362': 130,
    '11366': 131, '10312': 132, '10001': 133, '11426': 134, '10025': 135,
    '11413': 136, '11249': 137, '10019': 138, '10309': 139, '11411': 140,
    '11222': 141, '11697': 142, '11238': 143, '10044': 144, '10308': 145,
    '10307': 146, '10012': 147, '10464': 148, '11231': 149, '10128': 150,
    '10018': 151, '10075': 152, '10010': 153, '11001': 154, '11217': 155,
    '10021': 156, '10023': 157, '10017': 158, '10016': 159, '10014': 160,
    '10003': 161, '10013': 162, '10011': 163, '11040': 164, '11201': 165,
    '10028': 166, '10065': 167, '10024': 168, '10022': 169, '11215': 170,
    '11109': 171, '10280': 172, '10069': 173, '10005': 174, '10006': 175,
    '10004': 176, '10007': 177, '10282': 178,
}

bed_lookup = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
              20, 24, 30, 32, 35, 36, 40, 42, 50]

class Layer:
    def __init__(self, p_neurons, neurons, act) -> None:
        # Weights, biases, and activation function
        self.W = array(random.uniform(-1, 1, (p_neurons, neurons)))
        self.b = zeros((1, neurons))
        self.act = act

    def sig(x, deriv=False) -> ndarray:
        sig_x = 1 / (1 + exp(-x))

        if not deriv: return sig_x
        else: return sig_x * (1 - sig_x)
    
    def relu(x, deriv=False) -> ndarray:
        if not deriv: return where(x > 0, x, 0)
        else: return where(x > 0, 1, 0)

    def leaky_relu(x, deriv=False) -> ndarray:
        if not deriv: return where(x > 0, x, 0.01*x)
        else: return where(x > 0, 1.0, 0.01)
    
    def softmax(x, deriv=False) -> ndarray:
        exp_x = exp(x - npmax(x, axis=1, keepdims=True))
        soft_x = exp_x / npsum(exp_x, axis=1, keepdims=True)
        
        if not deriv: return soft_x
        else: return soft_x * (1 - soft_x)

class Nnet:
    def __init__(self, l_neurons=[6,3], l_acts=[Layer.sig, Layer.sig]):
        self.l_neurons = l_neurons
        self.l_acts = l_acts

    def crossent_grad(self, y, p) -> ndarray:
        p = clip(p, 1e-12, 1 - 1e-12)
        return - (y / p) + (1 - y) / (1 - p)
    
    def train(self, X: ndarray, y: ndarray, epochs=1000, alpha=0.01, batch_size=32):
        n_samples, n_feats = X.shape
        n_layers = len(self.l_neurons)

        layers = [Layer(n_feats, self.l_neurons[0], Layer.sig)]
        for i in range(1, n_layers):
            layers.append(Layer(self.l_neurons[i-1],
                                self.l_neurons[i],
                                self.l_acts[i]))

        out = Layer(self.l_neurons[-1], 27, Layer.softmax)

        accs = []

        for _ in range(epochs):
            loss = 0.0
            acc = 0.0

            for i in range(0, n_samples, batch_size):
                X_mini = X[i : i + batch_size]
                y_mini = y[i : i + batch_size]
                
                # Forward Pass
                l_in = [X_mini.dot(layers[0].W) + layers[0].b]
                l_out = [layers[0].act(l_in[0])]

                for j in range(1, n_layers):
                    l_in.append(l_out[j-1].dot(layers[j].W) + layers[j].b)
                    l_out.append(layers[j].act(l_in[j]))
                
                out_in = l_out[n_layers-1].dot(out.W) + out.b
                p = out.act(out_in)

                # Calc acc
                acc += self.acc_score(y_mini, p)

                # Back Prop
                out_grad = self.crossent_grad(y_mini, p) * out.act(out_in, True)
                out_W_grad = l_out[n_layers-1].T.dot(out_grad)
                out_b_grad = npsum(out_grad, axis=0, keepdims=True)

                l_grad = [out_grad.dot(out.W.T) *
                          layers[n_layers-1].act(l_in[n_layers-1], True)]
                l_W_grad = []
                l_b_grad = []
                for j in range(1, n_layers):
                    l_W_grad.insert(0, l_out[n_layers-j-1].T.dot(l_grad[0]))
                    l_b_grad.insert(0, npsum(l_grad[0], axis=0, keepdims=True))

                    l_grad.insert(0,
                                  l_grad[0].dot(layers[n_layers-j].W.T) *
                                  layers[n_layers-j-1].act(l_in[n_layers-j-1], True))

                l_W_grad.insert(0, X_mini.T.dot(l_grad[0]))
                l_b_grad.insert(0, npsum(l_grad[0], axis=0, keepdims=True))

                # Update weights and biases
                for j in range(0, n_layers):
                    layers[j].W -= alpha * l_W_grad[j]
                    layers[j].b -= alpha * l_b_grad[j]

                out.W -= alpha * out_W_grad
                out.b -= alpha * out_b_grad

            accs.append(acc / (n_samples / batch_size))

        self.layers = layers
        self.out = out
        
        return accs
    
    def predict(self, X: ndarray):
        n_layers = len(self.l_neurons)

        l_in = [X.dot(self.layers[0].W) + self.layers[0].b]
        l_out = [self.layers[0].act(l_in[0])]

        for i in range(1, n_layers):
            l_in.append(l_out[i-1].dot(self.layers[i].W) + self.layers[i].b)
            l_out.append(self.layers[i].act(l_in[i]))
                
        out_in = l_out[n_layers-1].dot(self.out.W) + self.out.b
        p = self.out.act(out_in)

        return p
    
    def acc_score(self, y, p):
        y_ind = argmax(y, axis=1)
        p_ind = argmax(p, axis=1)

        match_sum = 0
        for i in range(len(y_ind)):
            if y_ind[i] == p_ind[i]: match_sum += 1

        return match_sum / len(y_ind)


if __name__ == "__main__":
    def load_row(row: Dict[str,str]):
        type = row['TYPE'].replace('for sale', '')
        type = type.strip()
        type = type.lower()

        # Ranked based on mode, mean, and median beds of respective types
        home_type = 0
        if   type == 'co-op':             home_type = 0
        elif type == 'condo':             home_type = 1
        elif type == 'land':              home_type = 2
        elif type == 'pending':           home_type = 3
        elif type == 'contingent':        home_type = 4
        elif type == 'for sale':          home_type = 5
        elif type == 'house':             home_type = 6
        elif type == 'foreclosure':       home_type = 7
        elif type == 'townhouse':         home_type = 8
        elif type == 'coming soon':       home_type = 9
        elif type == 'mobile house':      home_type = 10
        elif type == 'multi-family home': home_type = 11

        co_op        = 1 if home_type == 0  else 0
        condo        = 1 if home_type == 1  else 0
        land         = 1 if home_type == 2  else 0
        pending      = 1 if home_type == 3  else 0
        contingent   = 1 if home_type == 4  else 0
        for_sale     = 1 if home_type == 5  else 0
        house        = 1 if home_type == 6  else 0
        foreclosure  = 1 if home_type == 7  else 0
        townhouse    = 1 if home_type == 8  else 0
        coming_soon  = 1 if home_type == 9  else 0
        mobile_house = 1 if home_type == 10 else 0
        multi_family = 1 if home_type == 11 else 0

        home_type_norm = home_type / 11

        price = int(row['PRICE'])
        price_norm = (price - 2494) / 2147481153

        bath = float(row['BATH'])
        bath_norm = bath / 50

        sqft = int(row['PROPERTYSQFT'])
        sqft_norm = (sqft - 230) / 65305

        addr = row['FORMATTED_ADDRESS']
        addr_parts = addr.split(',')

        zip_code     = addr_parts[len(addr_parts)-2].strip()
        zip_code     = zip_code.split(' ')[1]
        zip_code     = zip_code[0:5]
        zip_code_val = int(zip_code)
        zip_code_cat = zip_code_lookup[zip_code]
        
        zip_code_norm = (zip_code_cat - 1) / 177

        # Ranked by population density
        borough = 0
        if zip_code_val <= 10282:            borough = 0 # Manhattan
        elif 10301 <= zip_code_val <= 10314: borough = 4 # Staten Island
        elif 10451 <= zip_code_val <= 10457: borough = 2 # Bronx
        elif 11201 <= zip_code_val <= 11266: borough = 1 # Brooklyn
        else:                                borough = 3 # Queens

        manhattan     = 1 if borough == 0 else 0
        brooklyn      = 1 if borough == 1 else 0
        bronx         = 1 if borough == 2 else 0
        queens        = 1 if borough == 3 else 0
        staten_island = 1 if borough == 4 else 0

        borough_norm = borough / 4

        lat = float(row['LATITUDE'])
        long = float(row['LONGITUDE'])

        lat_norm = (lat - 40.4995462) / 0.4131833
        long_norm = (long + 74.2530332) / 0.5505832

        form_row = [
            # home_type_norm,
            
            house,
            townhouse,
            condo,
            multi_family,
            co_op,
            land,
            
            # contingent,
            # mobile_house,
            # for_sale,
            # foreclosure,
            # coming_soon,
            # pending,

            price_norm,
            bath_norm,
            sqft_norm,

            zip_code_norm,
            borough_norm,
            
            # manhattan,
            # staten_island,
            # bronx,
            # brooklyn,
            # queens,
            
            # lat_norm,
            long_norm
        ]

        return form_row
    
    def one_hot(y):
        mod_y = []
        for i in y:
            mod_row = zeros(27, dtype=int)
            mod_row[bed_lookup.index(i)] = 1
            mod_y.append(mod_row)
        return array(mod_y)

    train_data_file = 'train_data'
    train_label_file = 'train_label'
    test_data_file = 'test_data'

    if (len(sys.argv) > 1):
        train_data_file = 'testing/' + train_data_file + str(sys.argv[1])
        train_label_file = 'testing/' + train_label_file + str(sys.argv[1])
        test_data_file = 'testing/' + test_data_file + str(sys.argv[1])

    train_data_file += '.csv'
    train_label_file += '.csv'
    test_data_file += '.csv'

    train_data_dict = csv.DictReader(open(train_data_file, newline=''))

    train_data = []
    for row in train_data_dict:
        form_row = load_row(row)
        train_data.append(form_row)
    train_data = array(train_data)

    train_label_csv = csv.reader(open(train_label_file, newline=''))

    train_labels = []
    next(train_label_csv)
    for row in train_label_csv:
        train_labels.append(int(row[0]))
    train_labels = one_hot(train_labels)

    neural_net = Nnet([6, 3], [Layer.sig, Layer.sig])

    accs = neural_net.train(train_data, train_labels,
                            epochs=2500, alpha=0.001, batch_size=64)

    test_data_dict = csv.DictReader(open(test_data_file, newline=''))

    test_data = []
    for row in test_data_dict:
        form_row = load_row(row)
        test_data.append(form_row)
    test_data = array(test_data)

    res = neural_net.predict(test_data)
    res_val = argmax(res, axis=1)
    res_val = [bed_lookup[i] for i in res_val]
    resfile = DataFrame(res_val, columns=['BEDS'])
    resfile.to_csv('output.csv', index=False)

    if (len(sys.argv) > 1):
        test_label_csv = csv.reader(
            open('testing/test_label' + str(sys.argv[1]) + '.csv', newline=''))

        test_labels = []
        next(test_label_csv)
        for row in test_label_csv:
            test_labels.append(int(row[0]))
        test_labels = one_hot(test_labels)

        print(neural_net.acc_score(test_labels, res))

        plt.title('Learning Curve')
        plt.plot(range(len(accs)), accs)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()