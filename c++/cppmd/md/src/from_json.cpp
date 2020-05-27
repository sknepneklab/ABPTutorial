#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

#include "from_json.hpp"

void FastJsonReader::read_json_fast(const std::string &filename)
{
    particles.clear();

    using json = nlohmann::json;
    bool vertices_flag = false; //< check if everything was loaded
    bool faces_flag = false;
    // read a JSON file
    std::ifstream file_json(filename);
    if (file_json.good())
    {
        json json_data;
        file_json >> json_data;
        if (json_data["system"].find("box") != json_data["system"].end())
        {
            box.L.x = json_data["system"]["box"]["Lx"];
            box.L.y = json_data["system"]["box"]["Ly"];
            box.Llo.x = -0.5 * box.L.x;
            box.Lhi.x = 0.5 * box.L.x;
            box.Llo.y = -0.5 * box.L.y;
            box.Lhi.y = 0.5 * box.L.y;
            box.periodic.x = true;
            box.periodic.y = true;
        }
        else
        {
            std::cerr << "json box not found " << std::endl;
            exit(0);
        }
        if (json_data["system"].find("particles") != json_data["system"].end())
        {
            std::default_random_engine generator;
            std::uniform_real_distribution<double> uniform(-defPI, defPI);
            for (auto pf : json_data["system"]["particles"])
            {
                ParticleType p;
                p.id = pf["id"];
                p.radius = pf["radius"];
                p.r.x = pf["r"][0];
                p.r.y = pf["r"][1];
                double theta = uniform(generator);
                double nx = cos(theta);
                double ny = sin(theta);
                double vx = 0.0, vy = 0.0;
                double fx = 0.0, fy = 0.0;
                if (pf.find("v") != pf.end())
                {
                    vx = pf["v"][0];
                    vy = pf["v"][1];
                }
                if (pf.find("n") != pf.end())
                {
                    nx = pf["n"][0];
                    ny = pf["n"][1];
                }
                if (pf.find("f") != pf.end())
                {
                    fx = pf["f"][0];
                    fy = pf["f"][1];
                }
                p.forceC.x = fx;
                p.forceC.y = fy;
                p.v.x = vx;
                p.v.y = vy;
                p.n.x = nx;
                p.n.y = ny;
                particles.push_back(p);
            }
        }
        else
        {
            std::cerr << "json particles not found " << std::endl;
            exit(0);
        }
        std::cout << "Reading file " << filename << " success" << std::endl;
    }
    else  std::cerr << "Reading file " << filename << "<<<FAIL>>>" << std::endl;
}
