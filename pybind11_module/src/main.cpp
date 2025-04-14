#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
namespace py = pybind11;
#define rep(i,l,r) for (int i = l; i<=r; i++)
typedef long long ll;

py::dict bio2brat(const std::vector<int>& tags, const std::vector<double>& probs, const std::vector<int>& doc_ids, const std::vector<int>& start_pos, const std::vector<int>& tot_text_lens, bool use_bioe){

    py::dict all_doc_annos;
    int max_doc_id = 0;
    for (auto id:doc_ids){
        if (id > max_doc_id) max_doc_id = id;
    }
    rep(i,0,max_doc_id) all_doc_annos[pybind11::cast(i)] = new py::list;

    int last_j = -1;
    double score = 0, score_cnt = 0;
    rep(j,0,tags.size()){
        
        // start of new document
        if ((j > 0 && doc_ids[j] != doc_ids[j-1])|| j == tags.size()){
            int doc_id;
            if (j == tags.size()) doc_id = doc_ids[j];
                else doc_id = doc_ids[j-1];

            if (last_j != -1){
                py::list new_list;
                new_list.append(start_pos[last_j]);
                new_list.append(tot_text_lens[doc_id]);
                new_list.append(score / score_cnt);
                all_doc_annos[pybind11::cast(doc_id)].cast<py::list>().append(new_list);
            }
            score = 0;
            score_cnt = 0;
            last_j = -1;
        }

        if (j < tags.size()){
            if (last_j != -1){
                int end = -1;
                if (use_bioe && tags[j] == 3) end = j+1;
                    else if (tags[j] == 0 || tags[j] == 2) end = j;
                
                if (end != -1 and end < tags.size()){
                    py::list new_list;
                    new_list.append(start_pos[last_j]);
                    new_list.append(start_pos[end]);
                    new_list.append(score / score_cnt);
                    all_doc_annos[pybind11::cast(doc_ids[j])].cast<py::list>().append(new_list);
                    score = 0;
                    score_cnt = 0;
                    last_j = -1;
                }
            }

            // begin of entities
            if (tags[j] == 2){
                last_j = j;
                score += probs[j];
                score_cnt += 1;
            }

            // count number of Is
            if (tags[j] == 1){
                score += probs[j];
                score_cnt += 1;
            }
        }
    }

    return all_doc_annos;
}

PYBIND11_MODULE(_core, m){
    m.def("bio2brat", &bio2brat, "Convert BIO(or BIOE) tags to a document_list of anno_list of tuples containing predictions");
}