// Exact dual-vector GT. No OpenMP or third-party dependencies.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Vectors {
    uint32_t dim = 0;
    std::vector<float> data;
    size_t size() const { return data.size() / dim; }
    const float* row(size_t i) const { return data.data() + i * dim; }
};

Vectors read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open " + path);
    const auto bytes = in.tellg();
    in.seekg(0);
    Vectors v;
    if (!in.read(reinterpret_cast<char*>(&v.dim), 4) || v.dim == 0)
        throw std::runtime_error("Invalid dimension: " + path);
    const uint64_t stride = 4ULL * (1ULL + v.dim);
    if (bytes < 0 || uint64_t(bytes) % stride)
        throw std::runtime_error("Truncated fvecs: " + path);
    const size_t n = uint64_t(bytes) / stride;
    v.data.resize(n * v.dim);
    in.seekg(0);
    for (size_t i = 0; i < n; ++i) {
        uint32_t dim;
        if (!in.read(reinterpret_cast<char*>(&dim), 4) || dim != v.dim ||
            !in.read(reinterpret_cast<char*>(v.data.data() + i * v.dim), v.dim * 4ULL))
            throw std::runtime_error("Invalid record: " + path);
    }
    for (float x : v.data)
        if (!std::isfinite(x)) throw std::runtime_error("Nonfinite vector: " + path);
    return v;
}

double l2(const float* a, const float* b, uint32_t dim) {
    double sum = 0;
    for (uint32_t j = 0; j < dim; ++j) {
        const double delta = double(a[j]) - double(b[j]);
        sum += delta * delta;
    }
    return std::sqrt(sum); // sqrt EACH modality BEFORE alpha weighting.
}

struct Job {
    uint32_t begin, end; // Original base IDs in [begin, end).
    std::string path;
    std::vector<uint32_t> ids;
};

int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            std::cerr << "Usage: exact_gt base1 base2 query1 query2 alpha jobs.tsv k threads\n";
            return 2;
        }
        const uint32_t endian = 1;
        if (*reinterpret_cast<const char*>(&endian) != 1 || sizeof(float) != 4)
            throw std::runtime_error("Requires little-endian IEEE float32 host");
        auto b1 = read_fvecs(argv[1]), b2 = read_fvecs(argv[2]);
        auto q1 = read_fvecs(argv[3]), q2 = read_fvecs(argv[4]);
        auto alpha = read_fvecs(argv[5]);
        const size_t n = b1.size(), nq = q1.size();
        const int k_arg = std::stoi(argv[7]), threads_arg = std::stoi(argv[8]);
        if (k_arg <= 0 || threads_arg <= 0 || n > INT32_MAX || nq > INT32_MAX)
            throw std::runtime_error("Invalid k, threads or dataset size");
        const uint32_t k = uint32_t(k_arg);
        if (b2.size() != n || q2.size() != nq || b1.dim != q1.dim || b2.dim != q2.dim ||
            alpha.dim != 1 || alpha.size() != nq)
            throw std::runtime_error("Base/query/alpha shape mismatch");
        for (float a : alpha.data)
            if (a < 0 || a > 1) throw std::runtime_error("alpha outside [0,1]");
        std::ifstream job_file(argv[6]);
        if (!job_file) throw std::runtime_error("Cannot open jobs file");
        std::vector<Job> jobs;
        std::string line;
        while (std::getline(job_file, line)) {
            std::istringstream fields(line);
            Job job;
            if (!(fields >> job.begin >> job.end) || !std::getline(fields >> std::ws, job.path) ||
                job.path.empty() || job.end > n || job.begin >= job.end || job.end - job.begin < k)
                throw std::runtime_error("Invalid GT job: " + line);
            job.ids.resize(nq * k);
            jobs.push_back(std::move(job));
        }
        if (jobs.empty()) throw std::runtime_error("No GT jobs");
        std::cout << "Exact L2: " << n << " base, " << nq << " queries, " << jobs.size()
                  << " GT states, k=" << k << std::endl;
        std::atomic<size_t> next{0};
        const unsigned threads = std::min<size_t>(threads_arg, nq);
        auto worker = [&]() {
            std::vector<double> scores(n);
            std::vector<uint32_t> ranked(n);
            for (size_t qi; (qi = next.fetch_add(1)) < nq;) {
                const double a = alpha.data[qi];
                for (size_t id = 0; id < n; ++id)
                    scores[id] = a * l2(q1.row(qi), b1.row(id), b1.dim)
                               + (1.0 - a) * l2(q2.row(qi), b2.row(id), b2.dim);
                std::iota(ranked.begin(), ranked.end(), 0);
                std::sort(ranked.begin(), ranked.end(), [&](uint32_t x, uint32_t y) {
                    return scores[x] < scores[y] || (scores[x] == scores[y] && x < y);
                });
                // Reuse one full ranking for every active set; exact, no distance matrix cache.
                for (auto& job : jobs) {
                    uint32_t found = 0;
                    for (uint32_t id : ranked) {
                        if (id >= job.begin && id < job.end) {
                            job.ids[qi * k + found++] = id;
                            if (found == k) break;
                        }
                    }
                }
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
        for (const auto& job : jobs) {
            std::ofstream out(job.path, std::ios::binary);
            for (size_t qi = 0; qi < nq; ++qi) {
                out.write(reinterpret_cast<const char*>(&k), 4);
                out.write(reinterpret_cast<const char*>(job.ids.data() + qi * k), k * 4ULL);
            }
            out.close();
            if (!out) throw std::runtime_error("Cannot write GT: " + job.path);
        }
        std::cout << "All GT states written." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
