#include <iostream>
#include <gl/glew.h>
#include <gl/freeglut.h>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <Magick++.h>
#include <Magick++/Image.h>
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768
GLint success;

const int MAX_POINT_LIGHTS = 3;
const int MAX_SPOT_LIGHTS = 2;
static float scale = 0.0f;
static float scale2 = 0.0f;
using namespace std;
using namespace glm;

static const char* pVS = "                                                      \n\
     #version 330                                                                   \n\
    layout (location = 0) in vec3 pos;                                             \n\
    layout (location = 1) in vec2 tex;                                             \n\
    layout (location = 2) in vec3 norm;                                            \n\
    uniform mat4 gWVP;                                                             \n\
    uniform mat4 gWorld;                                                           \n\
	out vec3 pos0;																	\n\
    out vec2 tex0;                                                                 \n\
    out vec3 norm0;                                                                \n\
    void main()                                                                    \n\
    {                                                                              \n\
        gl_Position = gWVP * vec4(pos, 1.0);                                       \n\
        tex0 = tex;                                                                \n\
        norm0 = (gWorld * vec4(norm, 0.0)).xyz;                                    \n\
		pos0 = (gWorld * vec4(pos, 1.0)).xyz;										\n\
    }";

static const char* pFS = "                                                         \n\
    #version 330                                                                    \n\
	const int MAX_POINT_LIGHTS = 3;													\n\
	const int MAX_SPOT_LIGHTS = 2;													\n\
    in vec2 tex0;                                                                   \n\
	in vec3 norm0;																	\n\
	in vec3 pos0;																	\n\
    out vec4 fragcolor;                                                             \n\
    struct BaseLight                                                                    \n\
	{                                                                                   \n\
		vec3 Color;                                                                     \n\
		float AmbientIntensity;                                                         \n\
		float DiffuseIntensity;                                                         \n\
	};                                                                                  \n\
	struct DirectionalLight                                                             \n\
	{                                                                                   \n\
		BaseLight Base;																	\n\
		vec3 Direction;                                                                 \n\
	};                                                                                  \n\
	struct Attenuation                                                                  \n\
	{                                                                                   \n\
		float Constant;                                                                 \n\
		float Linear;                                                                   \n\
		float Exp;                                                                      \n\
	};                                                                                  \n\
	struct PointLight                                                                    \n\
	{                                                                                    \n\
		 BaseLight Base;                                                                  \n\
		 vec3 Position;																		 \n\
		 Attenuation Atten;																	 \n\
	};																						\n\
	struct SpotLight                                                                         \n\
	{                                                                                           \n\
		PointLight Base;																		\n\
		vec3 Direction;                                                                         \n\
		float Cutoff;                                                                           \n\
	};                                                                                          \n\
	uniform int gNumPointLights;															\n\
	uniform int gNumSpotLights;																\n\
	uniform DirectionalLight gDirectionalLight;												\n\
	uniform PointLight gPointLights[MAX_POINT_LIGHTS];										\n\
	uniform SpotLight gSpotLights[MAX_SPOT_LIGHTS];											\n\
    uniform sampler2D gSampler;                                                             \n\
    uniform vec3 gEyeWorldPos;                                                              \n\
    uniform float gMatSpecularIntensity;                                                    \n\
    uniform float gSpecularPower;                                                           \n\
    vec4 CalcLightInternal(BaseLight Light, vec3 LightDirection, vec3 Normal){				\n\
        vec4 AmbientColor = vec4(Light.Color, 1.0f) * Light.AmbientIntensity;               \n\
        float DiffuseFactor = dot(Normal, -LightDirection);                                 \n\
                                                                                            \n\
        vec4 DiffuseColor  = vec4(0, 0, 0, 0);                                              \n\
        vec4 SpecularColor = vec4(0, 0, 0, 0);                                              \n\
                                                                                            \n\
        if (DiffuseFactor > 0) {                                                            \n\
            DiffuseColor = vec4(Light.Color, 1.0f) * Light.DiffuseIntensity * DiffuseFactor;\n\
                                                                                            \n\
            vec3 VertexToEye = normalize(gEyeWorldPos - pos0);                         \n\
            vec3 LightReflect = normalize(reflect(LightDirection, Normal));                 \n\
            float SpecularFactor = dot(VertexToEye, LightReflect);                          \n\
            SpecularFactor = pow(SpecularFactor, gSpecularPower);                           \n\
            if (SpecularFactor > 0) {                                                       \n\
                SpecularColor = vec4(Light.Color, 1.0f) *                                   \n\
                                gMatSpecularIntensity * SpecularFactor;                     \n\
            }                                                                               \n\
        }                                                                                   \n\
        return (AmbientColor + DiffuseColor + SpecularColor);                               \n\
    }                                                                                       \n\
                                                                                            \n\
    vec4 CalcDirectionalLight(vec3 Normal)                                                  \n\
    {                                                                                         \n\
        return CalcLightInternal(gDirectionalLight.Base, gDirectionalLight.Direction, Normal); \n\
    }                                                                                           \n\
                                                                                            \n\
    vec4 CalcPointLight(PointLight l, vec3 Normal)                                                \n\
    {                                                                                           \n\
        vec3 LightDirection = pos0 - l.Position;												\n\
        float Distance = length(LightDirection);                                                \n\
        LightDirection = normalize(LightDirection);                                             \n\
                                                                                            \n\
        vec4 Color = CalcLightInternal(l.Base, LightDirection, Normal);                         \n\
        float Attenuation =  l.Atten.Constant +                                                 \n\
                             l.Atten.Linear * Distance +                                        \n\
                             l.Atten.Exp * Distance * Distance;                                 \n\
                                                                                            \n\
        return Color / Attenuation;                                                             \n\
    }                                                                                           \n\
    vec4 CalcSpotLight(SpotLight l, vec3 Normal)                                         \n\
    {                                                                                           \n\
        vec3 LightToPixel = normalize(pos0 - l.Base.Position);                             \n\
        float SpotFactor = dot(LightToPixel, l.Direction);                                      \n\
                                                                                                \n\
        if (SpotFactor > l.Cutoff) {                                                            \n\
            vec4 Color = CalcPointLight(l.Base, Normal);                                        \n\
            return Color * (1.0 - (1.0 - SpotFactor) * 1.0/(1.0 - l.Cutoff));                   \n\
        }                                                                                       \n\
        else return vec4(0,0,0,0);                                                               \n\
	}                                                                                           \n\
    void main()                                                                                 \n\
    {                                                                                           \n\
        vec3 Normal = normalize(norm0);                                                       \n\
        vec4 TotalLight = CalcDirectionalLight(Normal);                                         \n\
                                                                                            \n\
        for (int i = 0 ; i < gNumPointLights ; i++) {                                           \n\
            TotalLight += CalcPointLight(gPointLights[i], Normal);                                            \n\
        }                                                                                       \n\
        for (int i = 0 ; i < gNumSpotLights ; i++) {                                            \n\
            TotalLight += CalcSpotLight(gSpotLights[i], Normal);                                \n\
        }                                                                                       \n\
                                                                                            \n\
        fragcolor = texture2D(gSampler, tex0.xy) * TotalLight;                             \n\
    }";

struct vertex {
	glm::vec3 m_pos;
	glm::vec2 m_tex;
	glm::vec3 m_norm;
	vertex(glm::vec3 inp1, glm::vec2 inp2) {
		m_pos = inp1;
		m_tex = inp2;
		m_norm = glm::vec3(0.0f, 0.0f, 0.0f);
	}
};
struct BaseLight
{
	glm::vec3 Color;
	float AmbientIntensity;
	float DiffuseIntensity;
	BaseLight()
	{
		Color = glm::vec3(0.0f, 0.0f, 0.0f);
		AmbientIntensity = 0.0f;
		DiffuseIntensity = 0.0f;
	}
};
struct DirectionalLight : public BaseLight
{
	glm::vec3 Direction;
	DirectionalLight()
	{
		Direction = glm::vec3(0.0f, 0.0f, 0.0f);
	}
};
struct PointLight : public BaseLight
{
	glm::vec3 Position;
	struct
	{
		float Constant;
		float Linear;
		float Exp;
	} Attenuation;
	PointLight()
	{
		Position = glm::vec3(0.0f, 0.0f, 0.0f);
		Attenuation.Constant = 1.0f;
		Attenuation.Linear = 0.0f;
		Attenuation.Exp = 0.0f;
	}
};
struct SpotLight : public PointLight
{
	glm::vec3 Direction;
	float Cutoff;

	SpotLight()
	{
		Direction = glm::vec3(0.0f, 0.0f, 0.0f);
		Cutoff = 0.0f;
	}
};
glm::mat4x4 RotMat(float RotateX, float RotateY, float RotateZ)
{
	float x = asin(RotateX);
	float y = asin(RotateY);
	float z = asin(RotateZ);

	glm::mat4x4 rx(1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, cosf(x), -sinf(x), 0.0f,
		0.0f, sinf(x), cosf(x), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	glm::mat4x4 ry(cosf(y), 0.0f, -sinf(x), 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		sinf(x), 0.0f, cosf(y), 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	glm::mat4x4 rz(cosf(z), -sinf(z), 0.0f, 0.0f,
		sinf(z), cosf(z), 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	return(rz * ry * rx);
}


class Pipeline
{
private:
	struct projection {
		float FOV;
		float Width;
		float Height;
		float zNear;
		float zFar;
	};
	struct camera {
		vec3 pos;
		vec3 target;
		vec3 up;
	};
	mat4 m = {
		m[0][0] = 1.0f, m[0][1] = 0.0f, m[0][2] = 0.0f, m[0][3] = 0.0f,
		m[1][0] = 0.0f, m[1][1] = 1.0f, m[1][2] = 0.0f, m[1][3] = 0.0f,
		m[2][0] = 0.0f, m[2][1] = 0.0f, m[2][2] = 1.0f, m[2][3] = 0.0f,
		m[3][0] = 0.0f, m[3][1] = 0.0f, m[3][2] = 0.0f, m[3][3] = 1.0f,
	};
	mat4 ScaleMat = m, RotateMat = m, TransMat = m, Proj = m, Cam = m, CamTrans = m;
	vec3 m_scale, m_trans, m_rot;
	projection myproj;
	camera mycam;
	mat4 WVP = m;
	mat4 World = m;

	vec3 cross(vec3 v1, vec3 v2) {
		float x = v1.y * v2.z - v1.z * v2.y;
		float y = v1.z * v2.x - v1.x * v2.z;
		float z = v1.x * v2.y - v1.y * v2.x;
		return vec3(x, y, z);
	}
	void norm(vec3& v) {
		float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
		v.x /= len;
		v.y /= len;
		v.z /= len;
	}
	void InitScaleTransform() {
		ScaleMat = m;
		ScaleMat[0][0] = m_scale.x;
		ScaleMat[1][1] = m_scale.y;
		ScaleMat[2][2] = m_scale.z;
	};
	void InitRotateTransform() {
		mat4 rx, ry, rz;
		rx = m;
		ry = m;
		rz = m;
		const float x = radians(m_rot.x);
		const float y = radians(m_rot.y);
		const float z = radians(m_rot.z);

		rx[1][1] = cosf(x); rx[1][2] = -sinf(x);
		rx[2][1] = sinf(x); rx[2][2] = cosf(x);

		ry[0][0] = cosf(y); ry[0][2] = -sinf(y);
		ry[2][0] = sinf(y); ry[2][2] = cosf(y);

		rz[0][0] = cosf(z); rz[0][1] = -sinf(z);
		rz[1][0] = sinf(z); rz[1][1] = cosf(z);

		RotateMat = rz * ry * rx;
	};
	void InitTranslationMatrix() {
		TransMat = m;
		TransMat[0][3] = m_trans.x;
		TransMat[1][3] = m_trans.y;
		TransMat[2][3] = m_trans.z;
	};
	void InitPerspective() {
		float ar = myproj.Width / myproj.Height;
		float zNear = myproj.zNear;
		float zFar = myproj.zFar;
		float zRange = zNear - zFar;
		float tanHalfFOV = tanf(radians(myproj.FOV / 2.0));

		Proj = m;
		Proj[0][0] = 1 / (tanHalfFOV * ar);
		Proj[1][1] = 1 / tanHalfFOV;
		Proj[2][2] = (-zNear - zFar) / zRange;
		Proj[2][3] = 2. * zFar * zNear / zRange;
		Proj[3][2] = 1.0f;
		Proj[3][3] = 0.0f;
	};
	void InitCamera() {
		vec3 n = mycam.target;
		vec3 u = mycam.up;
		norm(n);
		norm(u);
		u = cross(u, mycam.target);
		vec3 v = cross(n, u);
		Cam = m;
		Cam[0][0] = u.x; Cam[0][1] = u.y; Cam[0][2] = u.z;
		Cam[1][0] = v.x; Cam[1][1] = v.y; Cam[1][2] = v.z;
		Cam[2][0] = n.x; Cam[2][1] = n.y; Cam[2][2] = n.z;
	}
	void InitCamTrans() {
		CamTrans = m;
		CamTrans[0][3] = -mycam.pos.x;
		CamTrans[1][3] = -mycam.pos.y;
		CamTrans[2][3] = -mycam.pos.z;
	}

public:
	Pipeline() {
		m_scale = { 1.0f, 1.0f, 1.0f };
		m_trans = { 0.0f, 0.0f, 0.0f };
		m_rot = { 0.0f, 0.0f, 0.0f };
	}
	void Scale(float x, float y, float z) {
		m_scale = { x, y,z };
	}
	void WorldPos(float x, float y, float z) {
		m_trans = { x, y,z };
	}
	void Rotate(float x, float y, float z) {
		m_rot = { x, y,z };
	}
	void SetPerspectiveProj(float a, float b, float c, float d, float e) {
		myproj.FOV = a;
		myproj.Height = b;
		myproj.Width = c;
		myproj.zFar = d;
		myproj.zNear = e;
	}
	void SetCamera(vec3 pos, vec3 target, vec3 up) {
		mycam.pos = pos;
		mycam.target = target;
		mycam.up = up;
	}
	mat4* GetWVPTrans();
	mat4* GetWorldTrans();
};
mat4* Pipeline::GetWVPTrans()
{
	InitScaleTransform();
	InitRotateTransform();
	InitTranslationMatrix();

	WVP = ScaleMat * RotateMat * TransMat;
	return &WVP;
}
mat4* Pipeline::GetWorldTrans()
{
	GetWVPTrans();
	InitPerspective();
	InitCamera();
	InitCamTrans();

	World = WVP * CamTrans * Cam * Proj;
	return &World;
}

class Texture
{
public:
	Texture(GLenum TextureTarget, const std::string& FileName)
	{
		m_textureTarget = TextureTarget;
		m_fileName = FileName;
		m_pImage = NULL;
	}

	bool Load()
	{
		try {
			m_pImage = new Magick::Image(m_fileName);
			m_pImage->write(&m_blob, "RGBA");
		}
		catch (Magick::Error& Error) {
			cout << "Error loading texture '" << m_fileName << "': " << Error.what() << endl;
			return false;
		}

		glGenTextures(1, &m_textureObj);
		glBindTexture(m_textureTarget, m_textureObj);
		glTexImage2D(m_textureTarget, 0, GL_RGB, m_pImage->columns(), m_pImage->rows(), -0.5, GL_RGBA, GL_UNSIGNED_BYTE, m_blob.data());
		glTexParameterf(m_textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(m_textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		return true;
	}

	void Bind(GLenum TextureUnit)
	{
		glActiveTexture(TextureUnit);
		glBindTexture(m_textureTarget, m_textureObj);
	}

private:
	string m_fileName;
	GLenum m_textureTarget;
	GLuint m_textureObj;
	Magick::Image* m_pImage;
	Magick::Blob m_blob;
};
Texture* pTexture = NULL;
class Technique
{
public:

	Technique() { m_shaderProg = 0; };

	~Technique()
	{
		for (ShaderObjList::iterator it = m_shaderObjList.begin(); it != m_shaderObjList.end(); it++) {
			glDeleteShader(*it);
		}
		if (m_shaderProg != 0) {
			glDeleteProgram(m_shaderProg);
			m_shaderProg = 0;
		}
	}
	virtual bool Init()
	{
		m_shaderProg = glCreateProgram();

		if (m_shaderProg == 0) {
			GLchar InfoLog[1024];
			cerr << "Error creating shader program: " << InfoLog << endl;
			return false;
		}
		return true;
	}
	void Enable() { glUseProgram(m_shaderProg); };

protected:

	bool AddShader(GLenum ShaderType, const char* pShaderText)
	{
		GLuint ShaderObj = glCreateShader(ShaderType);

		if (ShaderObj == 0) {
			fprintf(stderr, "Error creating shader type %d\n", ShaderType);
			exit(0);
		}

		const GLchar* p[1];
		p[0] = pShaderText;

		glShaderSource(ShaderObj, 1, p, NULL);
		glCompileShader(ShaderObj);


		glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
		if (!success) {
			GLchar InfoLog[1024];
			glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
			fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
			exit(1);
		}

		glAttachShader(m_shaderProg, ShaderObj);
		return true;
	}


	bool Finalize()
	{
		GLint success;

		glLinkProgram(m_shaderProg);

		glGetProgramiv(m_shaderProg, GL_LINK_STATUS, &success);
		if (success == 0) {
			GLchar ErrorLog[1024];
			glGetProgramInfoLog(m_shaderProg, sizeof(ErrorLog), NULL, ErrorLog);
			fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
			return false;
		}
		for (ShaderObjList::iterator it = m_shaderObjList.begin(); it != m_shaderObjList.end(); it++) {
			glDeleteShader(*it);
		}
		m_shaderObjList.clear();

		return true;
	}

	GLint GetUniformLocation(const char* pUniformName)
	{
		GLint Location = glGetUniformLocation(m_shaderProg, pUniformName);

		if (Location == 0xFFFFFFFF) {
			cerr << "Warning! Unable to get the location of uniform " << pUniformName << endl;
		}
		return Location;
	};

private:

	GLuint m_shaderProg;
	typedef list<GLuint> ShaderObjList;
	ShaderObjList m_shaderObjList;
};

class LightingTechnique : public Technique {
public:
	LightingTechnique() {};

	virtual bool Init() {
		if (!Technique::Init()) return false;
		if (!AddShader(GL_VERTEX_SHADER, pVS)) return false;
		if (!AddShader(GL_FRAGMENT_SHADER, pFS)) return false;
		if (!Finalize())  return false;

		gWVPLocation = GetUniformLocation("gWVP");
		gWorldLocation = GetUniformLocation("gWorld");
		samplerLocation = GetUniformLocation("gSampler");

		dirLightColor = GetUniformLocation("gDirectionalLight.Base.Color");
		dirLightAmbientIntensity = GetUniformLocation("gDirectionalLight.Base.AmbientIntensity");
		dirLightDirection = GetUniformLocation("gDirectionalLight.Direction");
		dirLightDiffuseIntensity = GetUniformLocation("gDirectionalLight.Base.DiffuseIntensity");

		eyeWorldPosition = GetUniformLocation("gEyeWorldPos");
		matSpecularIntensityLocation = GetUniformLocation("gMatSpecularIntensity");
		matSpecularPowerLocation = GetUniformLocation("gSpecularPower");
		numPointLightsLocation = GetUniformLocation("gNumPointLights");
		numSpotLightsLocation = GetUniformLocation("gNumSpotLights");

		for (unsigned int i = 0; i < MAX_POINT_LIGHTS; i++) {
			char Name[128];
			memset(Name, 0, sizeof(Name));
			snprintf(Name, sizeof(Name), "gPointLights[%d].Base.Color", i);
			pointLights[i].Color = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gPointLights[%d].Base.AmbientIntensity", i);
			pointLights[i].AmbientIntensity = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gPointLights[%d].Position", i);
			pointLights[i].Position = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gPointLights[%d].Base.DiffuseIntensity", i);
			pointLights[i].DiffuseIntensity = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gPointLights[%d].Atten.Constant", i);
			pointLights[i].Atten.Constant = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gPointLights[%d].Atten.Linear", i);
			pointLights[i].Atten.Linear = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gPointLights[%d].Atten.Exp", i);
			pointLights[i].Atten.Exp = GetUniformLocation(Name);

			if (pointLights[i].Color == 0xFFFFFFFF ||
				pointLights[i].AmbientIntensity == 0xFFFFFFFF ||
				pointLights[i].Position == 0xFFFFFFFF ||
				pointLights[i].DiffuseIntensity == 0xFFFFFFFF ||
				pointLights[i].Atten.Constant == 0xFFFFFFFF ||
				pointLights[i].Atten.Linear == 0xFFFFFFFF ||
				pointLights[i].Atten.Exp == 0xFFFFFFFF) return false;
		}
		for (unsigned int i = 0; i < MAX_SPOT_LIGHTS; i++) {
			char Name[128];
			memset(Name, 0, sizeof(Name));
			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Base.Color", i);
			spotLights[i].Color = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Base.AmbientIntensity", i);
			spotLights[i].AmbientIntensity = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Position", i);
			spotLights[i].Position = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Direction", i);
			spotLights[i].Direction = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Cutoff", i);
			spotLights[i].Cutoff = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Base.DiffuseIntensity", i);
			spotLights[i].DiffuseIntensity = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Constant", i);
			spotLights[i].Atten.Constant = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Linear", i);
			spotLights[i].Atten.Linear = GetUniformLocation(Name);

			snprintf(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Exp", i);
			spotLights[i].Atten.Exp = GetUniformLocation(Name);

			if (spotLights[i].Color == 0xFFFFFFFF ||
				spotLights[i].AmbientIntensity == 0xFFFFFFFF ||
				spotLights[i].Position == 0xFFFFFFFF ||
				spotLights[i].Direction == 0xFFFFFFFF ||
				spotLights[i].Cutoff == 0xFFFFFFFF ||
				spotLights[i].DiffuseIntensity == 0xFFFFFFFF ||
				spotLights[i].Atten.Constant == 0xFFFFFFFF ||
				spotLights[i].Atten.Linear == 0xFFFFFFFF ||
				spotLights[i].Atten.Exp == 0xFFFFFFFF) return false;
		}
		if (dirLightAmbientIntensity == 0xFFFFFFFF ||
			gWorldLocation == 0xFFFFFFFF ||
			samplerLocation == 0xFFFFFFFF ||
			dirLightColor == 0xFFFFFFFF ||
			dirLightDirection == 0xFFFFFFFF ||
			dirLightDiffuseIntensity == 0xFFFFFFFF ||
			eyeWorldPosition == 0xFFFFFFFF ||
			matSpecularIntensityLocation == 0xFFFFFFFF ||
			matSpecularPowerLocation == 0xFFFFFFFF ||
			numPointLightsLocation == 0xFFFFFFFF ||
			numSpotLightsLocation == 0xFFFFFFFF) return false;

		return true;
	};

	void SetgWVP(const glm::mat4* gWorld) {
		glUniformMatrix4fv(gWVPLocation, 1, GL_TRUE, (const GLfloat*)gWorld);
	};
	void SetWorld(const glm::mat4* World)
	{
		glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, (const GLfloat*)World);
	}
	void SetTextureUnit(unsigned int unit) {
		glUniform1i(samplerLocation, unit);
	};
	void SetMatSpecularIntensity(float Intensity)
	{
		glUniform1f(matSpecularIntensityLocation, Intensity);
	}
	void SetMatSpecularPower(float Power)
	{
		glUniform1f(matSpecularPowerLocation, Power);
	}
	void SetEyeWorldPos(const glm::vec3& EyeWorldPos)
	{
		glUniform3f(eyeWorldPosition, EyeWorldPos.x, EyeWorldPos.y, EyeWorldPos.z);
	}
	void SetDirectionalLight(const DirectionalLight& Light) {
		glUniform3f(dirLightColor, Light.Color.x, Light.Color.y, Light.Color.z);
		glUniform1f(dirLightAmbientIntensity, Light.AmbientIntensity);
		glm::vec3 Direction = Light.Direction;
		normalize(Direction);
		glUniform3f(dirLightDirection, Direction.x, Direction.y, Direction.z);
		glUniform1f(dirLightDiffuseIntensity, Light.DiffuseIntensity);
	};
	void SetPointLights(unsigned int NumLights, const PointLight* Lights)
	{
		glUniform1i(numPointLightsLocation, NumLights);

		for (unsigned int i = 0; i < NumLights; i++) {
			glUniform3f(pointLights[i].Color, Lights[i].Color.x, Lights[i].Color.y, Lights[i].Color.z);
			glUniform1f(pointLights[i].AmbientIntensity, Lights[i].AmbientIntensity);
			glUniform1f(pointLights[i].DiffuseIntensity, Lights[i].DiffuseIntensity);
			glUniform3f(pointLights[i].Position, Lights[i].Position.x, Lights[i].Position.y, Lights[i].Position.z);
			glUniform1f(pointLights[i].Atten.Constant, Lights[i].Attenuation.Constant);
			glUniform1f(pointLights[i].Atten.Linear, Lights[i].Attenuation.Linear);
			glUniform1f(pointLights[i].Atten.Exp, Lights[i].Attenuation.Exp);
		}
	}
	void SetSpotLights(unsigned int NumLights, const SpotLight* Lights)
	{
		glUniform1i(numSpotLightsLocation, NumLights);

		for (unsigned int i = 0; i < NumLights; i++) {
			glUniform3f(spotLights[i].Color, Lights[i].Color.x, Lights[i].Color.y, Lights[i].Color.z);
			glUniform1f(spotLights[i].AmbientIntensity, Lights[i].AmbientIntensity);
			glUniform1f(spotLights[i].DiffuseIntensity, Lights[i].DiffuseIntensity);
			glUniform3f(spotLights[i].Position, Lights[i].Position.x, Lights[i].Position.y, Lights[i].Position.z);
			glm::vec3 Direction = Lights[i].Direction;
			Direction = normalize(Direction);
			glUniform3f(spotLights[i].Direction, Direction.x, Direction.y, Direction.z);
			glUniform1f(spotLights[i].Cutoff, cosf(glm::radians(Lights[i].Cutoff)));
			glUniform1f(spotLights[i].Atten.Constant, Lights[i].Attenuation.Constant);
			glUniform1f(spotLights[i].Atten.Linear, Lights[i].Attenuation.Linear);
			glUniform1f(spotLights[i].Atten.Exp, Lights[i].Attenuation.Exp);
		}
	}
private:
	GLuint gWVPLocation;
	GLuint gWorldLocation;
	GLuint samplerLocation;
	GLuint dirLightColor;
	GLuint dirLightAmbientIntensity;
	GLuint dirLightDirection;
	GLuint dirLightDiffuseIntensity;

	GLuint eyeWorldPosition;
	GLuint matSpecularIntensityLocation;
	GLuint matSpecularPowerLocation;

	GLuint numPointLightsLocation;
	GLuint numSpotLightsLocation;
	struct {
		GLuint Color;
		GLuint AmbientIntensity;
		GLuint DiffuseIntensity;
		GLuint Position;
		struct {
			GLuint Constant;
			GLuint Linear;
			GLuint Exp;
		} Atten;
	} pointLights[MAX_POINT_LIGHTS];
	struct {
		GLuint Color;
		GLuint AmbientIntensity;
		GLuint DiffuseIntensity;
		GLuint Position;
		GLuint Direction;
		GLuint Cutoff;
		struct {
			GLuint Constant;
			GLuint Linear;
			GLuint Exp;
		} Atten;
	} spotLights[MAX_SPOT_LIGHTS];
};

class ICallbacks
{
public:
	virtual void KeyboardCB(unsigned char Key, int x, int y) = 0;
	virtual void RenderSceneCB() = 0;
	virtual void IdleCB() = 0;
};

void GLUTBackendInit(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	Magick::InitializeMagick(*argv);
};

bool GLUTBackendCreateWindow(unsigned int Width, unsigned int Height, const char* name) {
	glutInitWindowSize(Width, Height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(name);
	GLenum res = glewInit();
	if (res != GLEW_OK) {
		cerr << "Error: " << glewGetErrorString(res) << endl;
		return false;
	}
	return true;
};
ICallbacks* ICall = NULL;
void RenderScene() {
	ICall->RenderSceneCB();
}
void Idle() {
	ICall->IdleCB();
}
void Keyboard(unsigned char Key, int x, int y) {
	ICall->KeyboardCB(Key, x, y);
}
void CB() {
	glutDisplayFunc(RenderScene);
	glutIdleFunc(Idle);
	glutKeyboardFunc(Keyboard);
}
void GLUTBackendRun(ICallbacks* p) {
	if (!p) {
		fprintf(stderr, "%s : callbacks not specified!\n", __FUNCTION__);
		return;
	}
	ICall = p;
	CB();
	glutMainLoop();
};
class Main : public ICallbacks
{
private:
	GLuint VBO;
	GLuint IBO;
	LightingTechnique* light;

	Texture* texture;
	DirectionalLight dirLight;
	void GenBuff() {
		vertex Vertices[4] = {
		vertex(glm::vec3(-0.2, -0.2, 0),glm::vec2(0,0)),
		vertex(glm::vec3(0.3, -0.2, 0.5),glm::vec2(0.5,0)),
		vertex(glm::vec3(0.3, -0.2, -0.5),glm::vec2(1,0)),
		vertex(glm::vec3(0, 0.4, 0),glm::vec2(0.5,1)),
		};


		unsigned int Indices[] = { 0, 3, 1,
							   1, 3, 2,
							   2, 3, 0,
							   0, 2, 1 };
		CalcNorm(Indices, 12, Vertices, 4);

		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW);

		glGenBuffers(1, &IBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);
	}
	void CalcNorm(const unsigned int* indices, unsigned int indcount, vertex* vertices, unsigned int vertcount) {
		for (unsigned int i = 0; i < indcount; i += 3) {
			unsigned int Index0 = indices[i];
			unsigned int Index1 = indices[i + 1];
			unsigned int Index2 = indices[i + 2];
			glm::vec3 v1 = vertices[Index1].m_pos - vertices[Index0].m_pos;
			glm::vec3 v2 = vertices[Index2].m_pos - vertices[Index0].m_pos;
			glm::vec3 norm = cross(v1, v2);
			normalize(norm);

			vertices[Index0].m_norm += norm;
			vertices[Index1].m_norm += norm;
			vertices[Index2].m_norm += norm;
		}
		for (unsigned int i = 0; i < vertcount; i++) {
			normalize(vertices[i].m_norm);
		}
	}
public:
	Main()
	{
		pTexture = NULL;
		light = NULL;
		dirLight.Color = glm::vec3(1.0f, 1.0f, 1.0f);
		dirLight.AmbientIntensity = 0.5;
		dirLight.DiffuseIntensity = 0.9f;
		dirLight.Direction = glm::vec3(0.0f, 0.0, -1.0);
	}
	~Main() {
		delete light;
		delete pTexture;
	};
	bool Init()
	{
		GenBuff();
		light = new LightingTechnique();
		if (!light->Init())
		{
			return false;
		}
		light->Enable();
		light->SetTextureUnit(0);

		pTexture = new Texture(GL_TEXTURE_2D, "test.png");

		if (!pTexture->Load()) {
			return false;
		}

		return true;
	}
	void Run()
	{
		GLUTBackendRun(this);
	}
	virtual void RenderSceneCB()
	{
		glClear(GL_COLOR_BUFFER_BIT); //очистка буфера кадра

		scale += 0.055f;
		scale2 += 0.55f;

		/*PointLight pl[3];
		pl[0].DiffuseIntensity = 0.5;
		pl[0].Color = glm::vec3(1.0f, 0.0f, 0.0f);
		pl[0].Position = glm::vec3(sinf(scale) * 10, 1.0f, cosf(scale) * 10);
		pl[0].Attenuation.Linear = 0.1f;

		pl[1].DiffuseIntensity = 0.5;
		pl[1].Color = glm::vec3(0.0f, 1.0f, 0.0f);
		pl[1].Position = glm::vec3(sinf(scale + 2.1f) * 10, 1.0f, cosf(scale + 2.1f) * 10);
		pl[1].Attenuation.Linear = 0.1f;

		pl[2].DiffuseIntensity = 0.5;
		pl[2].Color = glm::vec3(0.0f, 0.0f, 1.0f);
		pl[2].Position = glm::vec3(sinf(scale + 4.2f) * 10, 1.0f, cosf(scale + 4.2f) * 10);
		pl[2].Attenuation.Linear = 0.1f;*/


		Pipeline p;
		//p.Scale(0.0f, 0.0f, 0.0f);
		//p.WorldPos(0.0f, 0.0f, 0.0f);
		p.Rotate(0.0f, scale2, 0.0f);
		p.SetPerspectiveProj(30.0f, GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT, 1.0f, 100.0f);

		glm::vec3 CameraPos(0.0f, 0.0f, -5.0f);
		glm::vec3 CameraTarget(0.0f, 0.0f, 2.0f);
		glm::vec3 CameraUp(0.0f, 1.0f, 0.0f);
		p.SetCamera(CameraPos, CameraTarget, CameraUp);

		SpotLight sl[2];
		sl[0].DiffuseIntensity = 50;
		sl[0].Color = glm::vec3(1, 0, 0);
		sl[0].Position = glm::vec3(0, 0, -0.0f);
		sl[0].Direction = glm::vec3(sinf(scale), 0.0f, cosf(scale));
		sl[0].Attenuation.Linear = 0.1f;
		sl[0].Cutoff = 20;

		sl[1].DiffuseIntensity = 10;
		sl[1].Color = glm::vec3(0, 1, 0);
		sl[1].Position = -CameraPos;
		sl[1].Direction = -CameraTarget;
		sl[1].Attenuation.Linear = 0.1f;
		sl[1].Cutoff = 10;

		light->SetSpotLights(2, sl);

		PointLight pl[3];
		pl[0].DiffuseIntensity = 0.5;
		pl[0].Color = glm::vec3(1.0f, 0.0f, 0.0f);
		pl[0].Position = glm::vec3(sinf(scale) * 10, 1.0f, cosf(scale) * 10);
		pl[0].Attenuation.Linear = 0.1f;

		pl[1].DiffuseIntensity = 0.5;
		pl[1].Color = glm::vec3(0.0f, 1.0f, 0.0f);
		pl[1].Position = glm::vec3(sinf(scale + 2.1f) * 10, 1.0f, cosf(scale + 2.1f) * 10);
		pl[1].Attenuation.Linear = 0.1f;

		pl[2].DiffuseIntensity = 0.5;
		pl[2].Color = glm::vec3(0.0f, 0.0f, 1.0f);
		pl[2].Position = glm::vec3(sinf(scale + 4.2f) * 10, 1.0f, cosf(scale + 4.2f) * 10);
		pl[2].Attenuation.Linear = 0.1f;

		light->SetgWVP(p.GetWVPTrans());
		light->SetWorld(p.GetWorldTrans());
		light->SetDirectionalLight(dirLight);
		light->SetEyeWorldPos(CameraPos);
		light->SetMatSpecularIntensity(1.0f);
		light->SetMatSpecularPower(32);

		//light->SetPointLights(3, pl);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), 0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (const GLvoid*)12);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (const GLvoid*)20);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
		pTexture->Bind(GL_TEXTURE0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		glutSwapBuffers(); // меняем фоновый буфер и буфер кадра местами
	}
	virtual void IdleCB()
	{
		RenderSceneCB();
	}
	virtual void KeyboardCB(unsigned char Key, int x, int y)
	{
		switch (Key) {
		case 'q':
			glutLeaveMainLoop();
			break;

		case 'a':
			dirLight.AmbientIntensity += 0.05f;
			break;

		case 's':
			dirLight.AmbientIntensity -= 0.05f;
			break;

		case 'z':
			dirLight.DiffuseIntensity += 0.05f;
			break;

		case 'x':
			dirLight.DiffuseIntensity -= 0.05f;
			break;
		}
	}

};
int main(int argc, char** argv)
{
	GLUTBackendInit(argc, argv);

	if (!GLUTBackendCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "LABA 3")) {
		return 1;
	}

	Main* pApp = new Main();

	if (!pApp->Init()) return 1;

	pApp->Run();

	delete pApp;

	return 0;
}