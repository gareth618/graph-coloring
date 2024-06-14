#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <set>
#include <map>

#define pb push_back

using namespace std;
ifstream file("H:\\clion\\Graph_Coloring_GA\\binary\\flat1000_50_0.col");
//ifstream file("H:\\clion\\Graph_Coloring_GA\\instances\\queen14_14.col");
ofstream fout("out.txt");
//hyperparameters
int no_colors, best_no_colors;
int no_vertexes, no_edges;
int upper_bound_colors, lower_bound_colors;
int prev_best, rep;
vector<vector<bool>> graph;

int generations = 20000;
int population_size = 50;
int fitness_threshold = 4, current_best, best_overall;
double prob_mutation = 0.7;
uniform_real_distribution<double> uniformRealDistribution(0, 1);
random_device rd_real;
mt19937 gen_real(rd_real());

vector<vector<int>> population;
map<int, int> votes;

bool compare(vector<int> x, vector<int> y) {
    return x[0] < y[0];
}

void find_search_parameters();

void read_input();

void binary_search_no_colors();

bool find_solution();

void create_population();

void selection();

void evaluate(vector<int> &individual);

void crossover();

void get_parents(int &parent1, int &parent2);

void mutate_1(vector<int> &individual);

void mutate_2(int index);

void wisdomOfArtificialCrowds();

int main() {
    read_input();
    find_search_parameters();
    binary_search_no_colors();
    return 0;
}

void binary_search_no_colors() {
    best_no_colors = upper_bound_colors;
    while (lower_bound_colors <= upper_bound_colors) {
        no_colors = lower_bound_colors + (upper_bound_colors - lower_bound_colors) / 2;
        cout << no_colors << ' ';
        bool result = find_solution();

        if (result) {
            upper_bound_colors = no_colors - 1;
            cout << "GASIT PENTRU " << no_colors << '\n';
        } else {
            lower_bound_colors = no_colors + 1;
        }
    }
}

bool find_solution() {
    create_population();
    best_overall = 1e6;
    prev_best = best_overall;
    for (int generation = 0; generation < generations; generation++) {
        selection();
        if (current_best < best_overall) {
            best_overall = current_best;
        }
        //cout<<current_best<<' ';
        if (best_overall == 0) {
            break;
        }
        if (current_best > fitness_threshold) {
            //cout<<1<<'\n';
            while (population.size() < population_size) {
                crossover();
            }
        } else {   //cout<<2<<'\n';
            mutate_2(0);
        }
    }
    cout << best_overall << ' ';
    if (best_overall == 0)
        return true;
    wisdomOfArtificialCrowds();
    if (best_overall == 0)
        return true;
    return false;
}

void wisdomOfArtificialCrowds() {
    sort(population.begin(), population.end(), compare);
    for (int i = 1; i <= no_vertexes; i++) {
        for (int j = 1; j <= no_vertexes; j++) {
            if (graph[i][j] && population[0][i] == population[0][j]) {
                int max_votes = 0, color = -1;
                votes.clear();
                for (int k = 0; k < population_size / 2; k++) {
                    if (votes.find(population[k][i]) == votes.end()) {
                        votes.insert({population[k][i], 1});
                        if (max_votes == 0) {
                            max_votes = 1;
                            color = population[k][i];
                        }
                    } else {
                        votes.find(population[k][i])->second += 1;
                        if (max_votes < votes.find(population[k][i])->second) {
                            max_votes = votes.find(population[k][i])->second;
                            color = population[k][i];
                        }
                    }
                }
                if (color == -1) {
                    cout << "NU E BINE";
                } else {
                    population[0][i] = color;
                }
            }
        }
    }
    evaluate(population[0]);
    if (best_overall < population[0][0]) {
        best_overall = population[0][0];
    }
}

void mutate_2(int index) {
    for (int i = 1; i <= no_vertexes; i++) {
        for (int j = 1; j <= no_vertexes; j++) {
            if (graph[i][j] && population[index][i] == population[index][j]) {
                uniform_int_distribution<int> uniformIntDistribution(1, no_colors);
                random_device rd;
                mt19937 gen(rd());
                population[index][i] = uniformIntDistribution(gen);
                break;
            }
        }
    }
    evaluate(population[index]);
}

void crossover() {
    int parent1, parent2;
    int cut_point;
    vector<int> child;
    uniform_int_distribution<int> uniformIntDistribution(2, no_vertexes - 1);
    random_device rd;
    mt19937 gen(rd());

    get_parents(parent1, parent2);

    cut_point = uniformIntDistribution(gen);
    for (int i = 0; i <= cut_point; i++) {
        child.pb(population[parent1][i]);
    }
    for (int i = cut_point + 1; i <= no_vertexes; i++) {
        child.pb(population[parent2][i]);
    }
    if (uniformRealDistribution(gen_real) <= prob_mutation) {
        mutate_1(child);
    }
    evaluate(child);
    population.pb(child);


}

void mutate_1(vector<int> &individual) {
    for (int i = 1; i <= no_vertexes; i++) {
        set<int> adjacent_colors;
        for (int j = 1; j <= no_vertexes; j++) {
            if (graph[i][j]) {
                adjacent_colors.insert(individual[j]);
            }
        }
        if (adjacent_colors.find(individual[i]) != adjacent_colors.end()) {
            vector<int> valid_colors;
            for (int j = 1; j <= no_colors; j++) {
                if (adjacent_colors.find(j) == adjacent_colors.end()) {
                    valid_colors.pb(j);
                }
            }
            if (valid_colors.size() > 0) {
                uniform_int_distribution<int> uniformIntDistribution(0, valid_colors.size() - 1);
                random_device rd;
                mt19937 gen(rd());
                individual[i] = valid_colors[uniformIntDistribution(gen)];
            }
        }
    }
}

void get_parents(int &parent1, int &parent2) {
    int tp1, tp2;
    if (current_best > fitness_threshold) {
        uniform_int_distribution<int> uniformIntDistribution(0, population.size() - 1);
        random_device rd;
        mt19937 gen(rd());
        tp1 = uniformIntDistribution(gen);
        tp2 = uniformIntDistribution(gen);
        while (tp2 == tp1) {
            tp2 = uniformIntDistribution(gen);
        }
        if (population[tp1][0] < population[tp2][0]) {
            parent1 = tp1;
        } else {
            parent1 = tp2;
        }

        tp1 = uniformIntDistribution(gen);
        tp2 = uniformIntDistribution(gen);
        while (tp2 == tp1) {
            tp2 = uniformIntDistribution(gen);
        }
        if (population[tp1][0] < population[tp2][0]) {
            parent2 = tp1;
        } else {
            parent2 = tp2;
        }
    } else {
        parent1 = parent2 = 0;
    }
}


void create_population() {
    population.clear();
    uniform_int_distribution<int> uniformIntDistribution(1, no_colors);
    random_device rd;
    mt19937 gen(rd());
    for (int i = 0; i < population_size; i++) {
        vector<int> individual;
        individual.pb(-1);
        for (int j = 1; j <= no_vertexes; j++) {
            individual.pb(uniformIntDistribution(gen));
        }
        evaluate(individual);
        population.pb(individual);
    }


}

void evaluate(vector<int> &individual) {
    individual[0] = 0;
    for (int i = 1; i < no_vertexes; i++) {
        for (int j = i + 1; j <= no_vertexes; j++) {
            if (individual[i] == individual[j] && graph[i][j]) {
                individual[0]++;
            }
        }
    }
}

void find_search_parameters() {
    upper_bound_colors = no_vertexes;
    lower_bound_colors = 1;
}

void read_input() {
    int v1, v2;
    string vertex, edge;
    string line;

    getline(file, line);
    while(line[0]=='c')
    {
        getline(file, line);
    }
    cout<<line;
    stringstream iss(line);
    iss >> vertex >> edge >> no_vertexes >> no_edges;
    graph = vector<vector<bool>>(no_vertexes + 1, vector<bool>(no_vertexes + 1, false));
    while (getline(file, line)) {
        stringstream iss(line);
        iss >> edge >> v1 >> v2;
        graph[v1][v2] = graph[v2][v1] = true;
    }
}

void selection() {
    sort(population.begin(), population.end(), compare);
    current_best = population[0][0];
    if (current_best == prev_best) {
        rep++;
    } else {
        rep = 0;
    }
    prev_best = current_best;
    if (rep == 100) {
        rep = 0;
        while (population.size() > population_size / 4) {
            population.pop_back();
        }
        for (int i = 0; i <= population.size(); i++) {
            mutate_2(i);
        }

        uniform_int_distribution<int> uniformIntDistribution(1, no_colors);
        random_device rd;
        mt19937 gen(rd());
        while (population_size > population.size()) {
            vector<int> individual;
            individual.pb(-1);
            for (int j = 1; j <= no_vertexes; j++) {
                individual.pb(uniformIntDistribution(gen));
            }
            evaluate(individual);
            population.pb(individual);
        }

    }
    if (fitness_threshold < current_best) {
        while (population.size() > population_size / 2) {
            population.pop_back();
        }
    }

}

