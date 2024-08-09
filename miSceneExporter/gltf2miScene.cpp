// dependency assimp, fmt, tinyxml2
// convert gltf 2.0 to mitsuba scene
//
// 1. Material Support:
//      Only supports Principled BSDF.
//      Includes base color (color3D or texture), metallic, roughness, transmission, and index of refraction (IOR).
// 2. Lighting Support:
//      Only supports material emission (area light).
// 3. Animation Support:
//      Animated objects cannot have another animated object as their parent.

#include <filesystem>
#include <set>
#include <iostream>
#include <algorithm>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/importerdesc.h>
#include <assimp/Exporter.hpp>

#include <fmt/core.h>
#include <fmt/std.h>

#include <tinyxml2.h>

// c++ 17
namespace fs = std::filesystem;

const char* shapePrefix = "s";
const char* materialPrefix = "m";
const char* cameraPrefix = "c";


template<typename T, typename P>
inline bool isInSet(std::set<T>& set, P&& value) {
    return set.find(value) != set.end();
}

// Function to decode URL-encoded string
std::string decodeURL(const std::string& url) {
    auto decodeHexChar = [=](const std::string& str, int& i) {
        if (i + 2 < str.length()) {
            int high = std::stoi(str.substr(i + 1, 1), nullptr, 16);
            int low = std::stoi(str.substr(i + 2, 1), nullptr, 16);
            char decodedChar = static_cast<char>((high << 4) + low);
            i += 2; // Move past the encoded character
            return decodedChar;
        }
        return '%'; // Return '%' if not a valid encoding
        };

    std::ostringstream decoded;
    for (int i = 0; i < url.length(); ++i) {
        if (url[i] == '%') {
            decoded << decodeHexChar(url, i);
        }
        else {
            decoded << url[i];
        }
    }
    return decoded.str();
}

/*
    shape data structure
*/
class Shape
{
public:
    Shape(std::string& id, std::string& filePath, aiNode* node, aiMesh* mesh, aiMatrix4x4& to_world)
        :id(id), filePath(filePath), node(node), mesh(mesh), to_world(to_world)
    {}
    std::string id;
    std::string filePath;
    aiNode* node;
    aiMesh* mesh;
    aiMatrix4x4 to_world;

};

/*
    aiXXX structure to string
*/
inline std::string aiColor2String(const aiColor3D& color) {
    return fmt::format("{0} {1} {2}",
        color.r, color.g, color.b);
}

inline std::string aiVec2String(const aiVector3f& vec) {
    return fmt::format("{0} {1} {2}",
        vec.x, vec.y, vec.z);
}

inline std::string aiQuat2String(const aiQuaternion& quat) {
    return fmt::format("{0} {1} {2} {3}",
        quat.w, quat.x, quat.y, quat.z);
}

inline std::string aiMatrix2String(const aiMatrix4x4& matrix) {
    return fmt::format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}",
        matrix.a1, matrix.a2, matrix.a3, matrix.a4,
        matrix.b1, matrix.b2, matrix.b3, matrix.b4,
        matrix.c1, matrix.c2, matrix.c3, matrix.c4,
        matrix.d1, matrix.d2, matrix.d3, matrix.d4);
}

/*
    tinyxml function
*/
template <typename T>
void setNameValueAttribute(tinyxml2::XMLElement* element, const char* name, T&& value) {
    element->SetAttribute("name", name);
    element->SetAttribute("value", value);
}

template <typename T>
tinyxml2::XMLElement* createNameValueElement(tinyxml2::XMLDocument& doc, const char* type, const char* name, T&& value) {
    tinyxml2::XMLElement* element = doc.NewElement(type);
    setNameValueAttribute(element, name, value);
    return element;
}

tinyxml2::XMLElement* createTransformElement(tinyxml2::XMLDocument& doc, const char* name, aiMatrix4x4& matrix) {
    /*
      <transform name="">
            <matrix value="-0.0757886 0 -0.0468591 -1.95645 0 0.0891049 0 0.648205 0.0468591 0 -0.0757886 -1.77687 0 0 0 1" />
      </transform>
    */
    auto transformElement = doc.NewElement("transform");
    transformElement->SetAttribute("name", name);

    auto matrixElement = doc.NewElement("matrix");
    matrixElement->SetAttribute("value", aiMatrix2String(matrix).c_str());

    transformElement->InsertEndChild(matrixElement);

    return transformElement;
}

tinyxml2::XMLElement* createToWorldElement(tinyxml2::XMLDocument& doc, aiMatrix4x4& matrix) {
    return createTransformElement(doc, "to_world", matrix);
}
tinyxml2::XMLElement* createToWorldElement(tinyxml2::XMLDocument& doc, aiMatrix4x4&& matrix) {
    return createTransformElement(doc, "to_world", matrix);
}

tinyxml2::XMLComment* createComment(tinyxml2::XMLDocument& doc, const char* comment) {
    return doc.NewComment(fmt::format("\t{}\t", comment).c_str());
}

bool saveDocument(tinyxml2::XMLDocument& doc, std::string filePath) {
    tinyxml2::XMLError eResult = doc.SaveFile(filePath.c_str());
    if (eResult != tinyxml2::XML_SUCCESS) {
        fmt::println("Error saving file: code {}", (int)eResult);
        return false;
    }
    fmt::println("Save document: {}\n", filePath);
    return true;
}

/*
    export all mesh to .obj file
*/
void exportObj(fs::path& outputFolder, fs::path& meshesFolder, aiScene* scene,
    aiNode* node, aiMatrix4x4& parentMatrix, std::set<std::string>& exportedMesh, std::vector<Shape>& shapes) {

    aiMatrix4x4 transformMatrix = parentMatrix * node->mTransformation;
    aiNode* newNode = new aiNode;
    newNode->mName = node->mName;
    newNode->mNumMeshes = 1;
    newNode->mMeshes = new unsigned int[1];
    newNode->mMetaData = node->mMetaData;
    newNode->mName = "ROOT";

    scene->mRootNode = newNode;

    for (size_t i = 0; i < node->mNumMeshes; i++)
    {
        std::string shapeId = std::string(node->mName.C_Str());
        if (node->mNumMeshes > 1) // split mesh
            shapeId = fmt::format("{}_{}", shapeId, i);

        auto indexOfMesh = node->mMeshes[i];
        auto mesh = scene->mMeshes[indexOfMesh];
        const char* meshName = mesh->mName.C_Str();
        std::string relativeFilePath = fmt::format("{}/{}.obj", meshesFolder.string(), meshName);

        if (isInSet(exportedMesh, meshName) == false) {
            newNode->mMeshes[0] = node->mMeshes[i];

            // export mesh in local coordinate
            Assimp::Exporter exporter;
            if (exporter.Export(scene, "objnomtl", (outputFolder / relativeFilePath).string()) == AI_SUCCESS) {
                fmt::println("Shape {} export mesh to {}", shapeId.c_str(), relativeFilePath);
            }
            else {
                fmt::println("Error exporting: {}", exporter.GetErrorString());
            }
            exportedMesh.insert(meshName);
        }
        else{
            static bool messagePrinted = false;
            if (messagePrinted == false) {
                fmt::println("[Warning] Find meshes with the same name, the shape will share the same mesh.");
                messagePrinted = true;
            }
            fmt::println("Shape {} use duplicate mesh {}", shapeId.c_str(), relativeFilePath);
        }
        shapes.push_back(Shape(shapeId, relativeFilePath, node, mesh, transformMatrix));
    }

    newNode->mMetaData = nullptr;
    delete newNode;

    for (size_t i = 0; i < node->mNumChildren; i++)
    {
        exportObj(outputFolder, meshesFolder, scene, node->mChildren[i], transformMatrix, exportedMesh ,shapes);
    }
}

void exportSceneMeshes(fs::path& outputFolder, fs::path& meshesFolder, const aiScene* scene,
    aiNode* node, std::vector<Shape>& shapes) {

    // copy scene with useful information
    aiScene* newScene = new aiScene();
    newScene->mNumMeshes = scene->mNumMeshes;
    newScene->mMeshes = scene->mMeshes;
    newScene->mNumMaterials = scene->mNumMaterials;
    newScene->mMaterials = scene->mMaterials;
    newScene->mNumTextures = scene->mNumTextures;
    newScene->mTextures = scene->mTextures;
    newScene->mMetaData = scene->mMetaData;

    // export obj with mutable scene
    aiMatrix4x4 matrix;
    std::set<std::string> alreadyExport;
    exportObj(outputFolder, meshesFolder, newScene, scene->mRootNode, matrix, alreadyExport, shapes);

    newScene->mRootNode = nullptr;
    newScene->mNumMeshes = 0;
    newScene->mMeshes = nullptr;
    newScene->mNumMaterials = 0;
    newScene->mMaterials = nullptr;
    newScene->mNumTextures = 0;
    newScene->mTextures = nullptr;
    newScene->mMetaData = nullptr;
    delete newScene;

}

void miIntegrator(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* docRoot) {
    /*
         <integrator type="path">
           <integer name="max_depth" value="8"/>
         </integrator>
    */
    docRoot->InsertEndChild(createComment(doc, "Integrator"));
    auto integratorElement = doc.NewElement("integrator");
    integratorElement->SetAttribute("type", "path");

    auto depthElement = createNameValueElement(doc, "integer", "max_depth", 8);
    integratorElement->InsertEndChild(depthElement);

    docRoot->InsertEndChild(integratorElement);
}

/*
*  out: cameras
*/
void miSensor(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* docRoot,
    const aiScene* pScene, std::set<std::string>& cameras) {
    /*
         <sensor type="perspective">
             <transform name="to_world">
                 <matrix value="-0.137283 -0.0319925 -0.990015 4.05402 2.71355e-008 0.999478 -0.0322983 1.61647 0.990532 -0.00443408 -0.137213 -2.30652 0 0 0 1" />
             </transform>
         </sensor>
      */
    docRoot->InsertEndChild(createComment(doc, "Camera"));
    for (int i = 0; i < pScene->mNumCameras; i++) {
        auto camera = pScene->mCameras[i];
        auto cameraNode = pScene->mRootNode->FindNode(camera->mName);
        cameras.insert(camera->mName.C_Str());

        // remove scale term
        aiQuaternion rotate;
        aiVector3D position;
        aiVector3D scale;
        cameraNode->mTransformation.Decompose(scale, rotate, position);
        scale = aiVector3D(1.0);
        aiMatrix4x4 toWorldMatrix(scale, rotate, position);

        aiMatrix4x4 cameraMarix;
        camera->GetCameraMatrix(cameraMarix);

        auto cameraId = fmt::format("{}_{}", cameraPrefix, cameraNode->mName.C_Str());
        auto sensorElement = doc.NewElement("sensor");
        sensorElement->SetAttribute("type", "perspective");
        sensorElement->SetAttribute("id", cameraId.c_str());

        auto transformElement = createToWorldElement(doc, toWorldMatrix * cameraMarix);

        sensorElement->InsertEndChild(transformElement);

        docRoot->InsertEndChild(sensorElement);
    }
}

/* out: materials */
void miShape(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* docRoot, const aiScene* pScene,
    std::vector<Shape>& shapes, std::set<aiMaterial*>& materials) {
    /*
     <shape type = "obj">
        <string name = "filename" value = "meshes/Teapot2.obj" / >
        <ref id = "mat-Teapot2" name = "bsdf" / >
        <emitter type="area">
            <rgb name="radiance" value="7.59909, 7.59909, 7.59909" />
        </emitter>
        <transform name="to_world">
            <matrix value="-0.0757886 0 -0.0468591 -1.95645 0 0.0891049 0 0.648205 0.0468591 0 -0.0757886 -1.77687 0 0 0 1" />
        </transform>
     </shape>
     */


     // collect all emitter and output togather
    std::vector<tinyxml2::XMLElement*> emitters;

    docRoot->InsertEndChild(createComment(doc, "Shapes"));
    for (auto& shape : shapes)
    {
        auto material = pScene->mMaterials[shape.mesh->mMaterialIndex];
        materials.insert(material);

        auto shapeId = fmt::format("{}_{}", shapePrefix, shape.id.c_str());
        auto shapeElement = doc.NewElement("shape");
        shapeElement->SetAttribute("type", "obj");
        shapeElement->SetAttribute("id", shapeId.c_str());

        // shape
        auto meshElement = createNameValueElement(doc, "string", "filename", shape.filePath.c_str());
        shapeElement->InsertEndChild(meshElement);

        // to_world
        auto toWorldElement = createToWorldElement(doc, shape.to_world);
        shapeElement->InsertEndChild(toWorldElement);

        // material
        auto materialId = fmt::format("{}_{}", materialPrefix, material->GetName().C_Str());
        auto materialElement = doc.NewElement("ref");
        materialElement->SetAttribute("name", "bsdf");
        materialElement->SetAttribute("id", materialId.c_str());
        shapeElement->InsertEndChild(materialElement);

        // emitter
        aiColor3D color;
        float intensity = 0.0f;
        if (material->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS &&
            material->Get(AI_MATKEY_EMISSIVE_INTENSITY, intensity) == AI_SUCCESS)
        {
            auto emitterElement = doc.NewElement("emitter");
            emitterElement->SetAttribute("type", "area");

            auto colorElement = createNameValueElement(doc, "rgb", "radiance", aiColor2String(color * intensity).c_str());

            emitterElement->InsertEndChild(colorElement);
            shapeElement->InsertEndChild(emitterElement);

            // store to array and process next shape
            emitters.push_back(shapeElement);
            continue;
        }

        docRoot->InsertEndChild(shapeElement);
    }

    docRoot->InsertEndChild(createComment(doc, "Emitters"));
    for (auto& emitter : emitters) {
        docRoot->InsertEndChild(emitter);
    }
    if (emitters.size() == 0) {
        /*<emitter type="constant">
            <rgb name="radiance" value="1.0"/>
        </emitter>*/
        docRoot->InsertEndChild(createComment(doc, "There is no emitter in the scene. Create a constant environment emitter."));

        auto envElement = doc.NewElement("emitter");
        envElement->SetAttribute("type", "constant");
        auto element = createNameValueElement(doc, "rgb", "radiance", 3.0f);
        envElement->InsertEndChild(element);
        docRoot->InsertEndChild(envElement);
    }
}

/*  out: textures */
void miMaterial(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* docRoot, const aiScene* pScene, fs::path inputFile,
    std::set<aiMaterial*>& materials, std::set<fs::path>& textures) {
    /*
     <bsdf type="twosided" id="Floor">
        <bsdf type="principled" id="name">
            <rgb name="base_color" value="1.0,1.0,1.0"/>
            <texture type="bitmap" name="base_color">
                <string name="filename" value="wood.jpg"/>
            </texture>
            <float name="metallic" value="0.7" />
            <float name="specular" value="0.6" />
            <float name="roughness" value="0.2" />
            <float name="spec_trans" value="1"/>
        </bsdf>
     </bsdf>
    */

    // default value for gltf 2.0, if material do have this attribute
    // the value will be used.
    const float defaulMetallic = 1.0f;
    const float defaultRoughness = 1.0f;
    const float defaultIOR = 1.5f;

    docRoot->InsertEndChild(createComment(doc, "Materials"));
    for (auto& material : materials)
    {
        fmt::println("Find material {}", material->GetName().C_Str());

        auto materialElement = doc.NewElement("bsdf");
        materialElement->SetAttribute("type", "principled");

        aiColor3D color;
        aiString relativeTexturePath;
        if (material->GetTexture(aiTextureType_BASE_COLOR, 0, &relativeTexturePath) == AI_SUCCESS) {
            auto element = doc.NewElement("texture");
            element->SetAttribute("type", "bitmap");
            element->SetAttribute("name", "base_color");

            relativeTexturePath = aiString(decodeURL(relativeTexturePath.C_Str()));
            auto fileName = createNameValueElement(doc, "string", "filename", relativeTexturePath.C_Str());

            element->InsertEndChild(fileName);
            materialElement->InsertEndChild(element);

            fs::path texturePath = inputFile.parent_path() / relativeTexturePath.C_Str();
            textures.insert(texturePath);
        }
        else if (material->Get(AI_MATKEY_BASE_COLOR, color) == AI_SUCCESS)
        {
            auto element = createNameValueElement(doc, "rgb", "base_color", aiColor2String(color).c_str());
            materialElement->InsertEndChild(element);
        }

        // the defualt value of metallic in gltf is 1.0 ,miScene default value is 0.
        // so always write metallic to miScene.
        float metallic = defaulMetallic;
        material->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
        {
            auto element = createNameValueElement(doc, "float", "metallic", metallic);
            materialElement->InsertEndChild(element);
        }

        // the defualt value of roughness in gltf is 1.0 , miScene default value is 0.5.
        // so always write roughness to miScene.
        float roughness = defaultRoughness;
        material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
        {
            auto element = createNameValueElement(doc, "float", "roughness", roughness);
            materialElement->InsertEndChild(element);
        }

        bool isTransmission = false;
        // the defualt value of transmissionFactor in gltf is 0.0, ior is 1.5
        // treat material is Transmission if AI_MATKEY_TRANSMISSION_FACTOR has value
        float transmissionFactor;
        float ior = defaultIOR;
        if (material->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor) == AI_SUCCESS)
        {
            isTransmission = true;

            material->Get(AI_MATKEY_REFRACTI, ior);

            auto element = createNameValueElement(doc, "float", "spec_trans", transmissionFactor);
            materialElement->InsertEndChild(element);

            element = createNameValueElement(doc, "float", "eta", ior);
            materialElement->InsertEndChild(element);
        }

        int isDoubleSided = false;
        material->Get(AI_MATKEY_TWOSIDED, isDoubleSided);

        auto materialId = fmt::format("{}_{}", materialPrefix, material->GetName().C_Str());
        if (isDoubleSided && isTransmission == false) {
            auto element = doc.NewElement("bsdf");
            element->SetAttribute("type", "twosided");
            element->SetAttribute("id", materialId.c_str());
            element->InsertEndChild(materialElement);

            docRoot->InsertEndChild(element);
        }
        else {
            materialElement->SetAttribute("id", materialId.c_str());
            docRoot->InsertEndChild(materialElement);
        }
    }
}

void copyTextures(std::set<fs::path>& textures, fs::path& outputFolder, fs::path& textureFolder) {
    // copy textures
    for (auto& texturePath : textures) {
        if (fs::exists(texturePath)) {
            auto destination = textureFolder / texturePath.filename();
            fs::copy(texturePath,
                outputFolder / destination,
                fs::copy_options::update_existing);
            fmt::println("Copy texture {} to {}", texturePath.string(), destination.string());
        }
        else
        {
            fmt::println("Texture {} not found ", texturePath.string());
        }
    }
}

void setAnimatedParent(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* docRoot,
    std::string& animatedParent, aiNode* node) {
    /*
        <shape id="s_Handle">
            <ref name="shape" value="s_Door"/>
        </shape>
    */
    for (size_t i = 0; i < node->mNumChildren; i++)
    {
        auto child = node->mChildren[i];

        for (size_t i_mesh = 0; i_mesh < child->mNumMeshes; i_mesh++)
        {
            auto shapeId = fmt::format("{}_{}", shapePrefix, child->mName.C_Str());
            if (child->mNumMeshes > 1)
                shapeId = fmt::format("{}_{}", shapeId, i_mesh);
            auto shapeElement = doc.NewElement("shape");
            shapeElement->SetAttribute("id", shapeId.c_str());
            auto refElement = createNameValueElement(doc, "ref", "shape", animatedParent.c_str());
            shapeElement->InsertEndChild(refElement);

            docRoot->InsertEndChild(shapeElement);
        }

        setAnimatedParent(doc, docRoot, animatedParent, node->mChildren[i]);
    }
}

/*
    create a new animation xml file
*/
void exportAnimation(tinyxml2::XMLDocument& doc, tinyxml2::XMLElement* docRoot,
    const aiScene* scene, std::set<std::string>& cameras) {
    /*
    <shape id="s_Door" max="177">
        <transform name="to_world">
            <matrix value="1 0 0 -2.0514352 0 1 0 -1.0226166 0 0 1 -0.81895125 0 0 0 1"/>
        </transform>
        <key value="000" time="0">
            <transform position="2.8494449 0.8184086 0.91905123" rotation="0.95741737 0 -0.2887074 -0" scaling="1 1 1"/>
        </key>
    </shape>
    <shape id="s_Handle">
        <ref name="shape" value="s_Door"/>
    </shape>
    */
    for (size_t i = 0; i < scene->mNumAnimations; i++)
    {
        auto animation = scene->mAnimations[i];
        for (size_t o = 0; o < animation->mNumChannels; o++)
        {
            auto channel = animation->mChannels[o];
            int step = std::max({ channel->mNumPositionKeys, channel->mNumRotationKeys, channel->mNumScalingKeys });
            int width = (int)std::ceil(std::log10(step));
            if (channel->mNumPositionKeys > 1 && channel->mNumPositionKeys != step ||
                channel->mNumRotationKeys > 1 && channel->mNumRotationKeys != step ||
                channel->mNumScalingKeys > 1 && channel->mNumScalingKeys != step) {
                fmt::println("Cannot handle keys different sizes pos:{}, rot:{}, scale{}", channel->mNumPositionKeys, channel->mNumRotationKeys, channel->mNumScalingKeys);
                continue;
            }

            auto node = scene->mRootNode->FindNode(channel->mNodeName);
            fmt::println("Find animated object: {}", node->mName.C_Str());

            auto prefix = isInSet(cameras, node->mName.C_Str()) ? cameraPrefix : shapePrefix;
            auto shapeId = fmt::format("{}_{}", prefix, channel->mNodeName.C_Str());
            if (node->mNumMeshes > 1)
                shapeId = fmt::format("{}_{}", shapeId, 0);
            auto shapeElement = doc.NewElement("shape");
            shapeElement->SetAttribute("id", shapeId.c_str());
            shapeElement->SetAttribute("max", step);
            if (node->mParent != scene->mRootNode) {
                aiMatrix4x4 m;
                auto p = node;
                while (p->mParent != nullptr)
                {
                    m = p->mParent->mTransformation * m;
                    p = p->mParent;
                }
                auto toWorld = createToWorldElement(doc, m);
                shapeElement->InsertEndChild(toWorld);
            }

            for (size_t i_key = 0; i_key < step; i_key++)
            {
                auto transformElement = doc.NewElement("transform");
                double time = 0.0;
                if (channel->mNumPositionKeys > 1 || i_key == 0) {
                    auto position = channel->mPositionKeys[i_key];
                    transformElement->SetAttribute("position", aiVec2String(position.mValue).c_str());
                    time = position.mTime;
                }
                if (channel->mNumRotationKeys > 1 || i_key == 0) {
                    auto rotation = channel->mRotationKeys[i_key];
                    transformElement->SetAttribute("rotation", aiQuat2String(rotation.mValue).c_str());
                    time = rotation.mTime;
                }
                if (channel->mNumScalingKeys > 1 || i_key == 0) {
                    auto scaling = channel->mScalingKeys[i_key];
                    transformElement->SetAttribute("scaling", aiVec2String(scaling.mValue).c_str());
                    time = scaling.mTime;
                }

                auto timeElement = doc.NewElement("key");
                timeElement->SetAttribute("value", fmt::format("{:0{}}", i_key, width).c_str());
                timeElement->SetAttribute("time", time);

                timeElement->InsertEndChild(transformElement);
                shapeElement->InsertEndChild(timeElement);
            }

            docRoot->InsertEndChild(shapeElement);

            for (size_t i_mesh = 1; i_mesh < node->mNumMeshes; i_mesh++)
            {
                auto otherShapeId = fmt::format("{}_{}_{}", shapePrefix, node->mName.C_Str(), i_mesh);
                auto shapeElement = doc.NewElement("shape");
                shapeElement->SetAttribute("id", otherShapeId.c_str());
                auto refElement = createNameValueElement(doc, "ref", "shape", shapeId.c_str());
                shapeElement->InsertEndChild(refElement);

                docRoot->InsertEndChild(shapeElement);
            }

            setAnimatedParent(doc, docRoot, shapeId, node);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        fmt::println("No input file. Try default input file path.");
    }

    fs::path inputFile = (argc > 1) ? argv[1] : "./data/untitled.gltf";
    fs::path outputFolder = "./miScene";
    fs::path meshesFolder = "models";
    fs::path textureFolder = "textures";
    const char* outputSceneFileName = "scene.xml";
    const char* outputAnimationFileName = "animation.xml";

    uint32_t assimpFlags = aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_RemoveComponent;

    assimpFlags &= ~(aiProcess_CalcTangentSpace);
    assimpFlags &= ~(aiProcess_FindDegenerates);
    assimpFlags &= ~(aiProcess_OptimizeGraph);
    assimpFlags &= ~(aiProcess_SplitLargeMeshes);

    Assimp::Importer importer;
    auto pScene = importer.ReadFile(inputFile.string(), assimpFlags);
    if (pScene == nullptr) {
        fmt::println("Cannot open file {} : {}", inputFile.string(), importer.GetErrorString());
        return -1;
    }

    fmt::println("Open {} with {}", inputFile.string(), importer.GetImporterInfo(importer.GetPropertyInteger("importerIndex"))->mName);

    // create directories
    fs::create_directories(outputFolder);
    fs::create_directories(outputFolder / meshesFolder);
    fs::create_directories(outputFolder / textureFolder);

    // create xml file
    tinyxml2::XMLDocument doc;
    auto docRoot = doc.NewElement("scene");
    docRoot->SetAttribute("version", "3.0.0");
    doc.InsertEndChild(docRoot);

    std::vector<Shape> shapes;
    std::set<aiMaterial*> materials;
    std::set<fs::path> textures;

    std::set<std::string> cameras;

    exportSceneMeshes(outputFolder, meshesFolder, pScene, pScene->mRootNode, shapes);

    miIntegrator(doc, docRoot);
    miSensor(doc, docRoot, pScene, cameras);
    miShape(doc, docRoot, pScene, shapes, materials);
    miMaterial(doc, docRoot, pScene, inputFile, materials, textures);

    copyTextures(textures, outputFolder, textureFolder);

    auto outputScenePath = outputFolder.string() + "/" + outputSceneFileName;
    if (saveDocument(doc, outputScenePath) == false) {
        return -1;
    }

    if (pScene->mNumAnimations > 0) {
        tinyxml2::XMLDocument aniDoc;
        auto aniDocRoot = aniDoc.NewElement("animation");
        aniDoc.InsertEndChild(aniDocRoot);

        exportAnimation(aniDoc, aniDocRoot, pScene, cameras);

        auto outputAniPath = outputFolder.string() + "/" + outputAnimationFileName;
        if (saveDocument(aniDoc, outputAniPath) == false) {
            return -1;
        }
    }

    return 0;
}