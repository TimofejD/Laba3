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

static float scale = 0.0f;
using namespace std;
using namespace glm;

static const char* pVS = "                                                      \n\
     #version 330                                                                   \n\
    layout (location = 0) in vec3 pos;                                             \n\
    layout (location = 1) in vec2 tex;                                             \n\
    layout (location = 2) in vec3 norm;                                            \n\
    uniform mat4 gWVP;                                                             \n\
    uniform mat4 gWorld;                                                           \n\
    out vec2 tex0;                                                                 \n\
    out vec3 norm0;                                                                \n\
    void main()                                                                    \n\
    {                                                                              \n\
        gl_Position = gWVP * vec4(pos, 1.0);                                       \n\
        tex0 = tex;                                                                \n\
        norm0 = (gWorld * vec4(norm, 0.0)).xyz;                                    \n\
    }";

static const char* pFS = "                                                         \n\
    #version 410                                                                    \n\
    in vec2 tex0;                                                                   \n\
	in vec3 norm0;																	\n\
    struct DirectionalLight                                                         \n\
    {                                                                               \n\
        vec3 Color;                                                                 \n\
        float AmbientIntensity;                                                     \n\
		vec3 Direction;                                                             \n\
        float DiffuseIntensity;                                                     \n\
    };                                                                              \n\
    uniform sampler2D gSampler;                                                     \n\
    uniform DirectionalLight gDirectionalLight;                                     \n\
    out vec4 fragcolor;                                                             \n\
    void main()                                                                     \n\
    {                                                                               \n\
        vec4 AmbientColor = vec4(gDirectionalLight.Color, 1.0f) *                   \n\
                        gDirectionalLight.AmbientIntensity;                         \n\
                                                                                    \n\
        float DiffuseFactor = dot(normalize(norm0), -gDirectionalLight.Direction);  \n\
                                                                                    \n\
        vec4 DiffuseColor;                                                          \n\
                                                                                    \n\
        if (DiffuseFactor > 0){                                                     \n\
            DiffuseColor = vec4(gDirectionalLight.Color, 1.0f) *                    \n\
                       gDirectionalLight.DiffuseIntensity *                         \n\
                       DiffuseFactor;                                               \n\
        }                                                                           \n\
        else{                                                                       \n\
            DiffuseColor = vec4(0,0,0,0);                                           \n\
        }                                                                           \n\
                                                                                    \n\
        fragcolor = texture2D(gSampler, tex0.xy) *                             \n\
                    (AmbientColor + DiffuseColor);                                  \n\
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
struct DirectionLight
{
	glm::vec3 Color;
	float AmbientIntensity;
	glm::vec3 Direction;
	float DiffuseIntensity;
};

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
private:
	GLuint gWVPLocation;
	GLuint gWorldLocation;
	GLuint samplerLocation;
	GLuint LightColor;
	GLuint LightAmbientIntensity;
	GLuint LightDirection;
	GLuint LightDiffuseIntensity;
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
		LightColor = GetUniformLocation("gDirectionalLight.Color");
		LightAmbientIntensity = GetUniformLocation("gDirectionalLight.AmbientIntensity");
		LightDirection = GetUniformLocation("gDirectionalLight.Direction");
		LightDiffuseIntensity = GetUniformLocation("gDirectionalLight.DiffuseIntensity");
		if (LightAmbientIntensity == 0xFFFFFFFF || gWorldLocation == 0xFFFFFFFF || samplerLocation == 0xFFFFFFFF || LightColor == 0xFFFFFFFF || LightDirection == 0xFFFFFFFF || LightDiffuseIntensity == 0xFFFFFFFF)  return false;

		return true;
	};

	void SetgWVP(const mat4* gWorld) {
		glUniformMatrix4fv(gWVPLocation, 1, GL_TRUE, (const GLfloat*)gWorld);
	};

	void SetWorld(const mat4* World)
	{
		glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, (const GLfloat*)World);
	}

	void SetTextureUnit(unsigned int unit) {
		glUniform1i(samplerLocation, unit);
	};

	void SetDirectionalLight(const DirectionLight& Light) {
		glUniform3f(LightColor, Light.Color.x, Light.Color.y, Light.Color.z);
		glUniform1f(LightAmbientIntensity, Light.AmbientIntensity);
		vec3 Direction = Light.Direction;
		normalize(Direction);
		glUniform3f(LightDirection, Direction.x, Direction.y, Direction.z);
		glUniform1f(LightDiffuseIntensity, Light.DiffuseIntensity);
	};
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
	DirectionLight dirLight;
	void GenBuff() {
		vertex Vertices[4]{
		vertex(vec3(-0.2, -0.2, 0),vec2(0,0)),
		vertex(vec3(0.3, -0.2, 0.5),vec2(0.5,0)),
		vertex(vec3(0.3, -0.2, -0.5),vec2(1,0)),
		vertex(vec3(0, 0.4, 0),vec2(0.5,1)),
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
		dirLight.AmbientIntensity = 0.5f;
		dirLight.DiffuseIntensity = 0.75f;
		dirLight.Direction = glm::vec3(1.0f, 0.0, 0.0);
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

		scale += 0.65f;
		Pipeline p;
		p.Scale(1.0f, 1.0f, 1.0f);
		p.WorldPos(0.0f, 0.0f, 0.0f);
		p.Rotate(0.0f, scale, 0.0f);
		p.SetPerspectiveProj(30.0f, GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT, 1.0f, 100.0f);

		glm::vec3 CameraPos(0.0f, 0.0f, -20.0f);
		glm::vec3 CameraTarget(0.0f, 5.0f, 2.0f);
		glm::vec3 CameraUp(0.0f, 1.0f, 0.0f);
		p.SetCamera(CameraPos, CameraTarget, CameraUp);

		light->SetgWVP(p.GetWVPTrans());
		light->SetWorld(p.GetWorldTrans());
		light->SetDirectionalLight(dirLight);

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