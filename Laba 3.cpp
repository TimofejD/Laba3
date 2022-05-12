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

GLuint VBO;
GLuint IBO;
GLuint gWorldLocation;
GLuint gSampler;

static float scale = 0.0f;
using namespace std;

//const char* pVS = "#version 330\n layout (location = 0) in vec3 Position; layout (location = 1) in vec2 TexCoord; uniform mat4 gWorld; out vec2 TexCoord0; void main() {gl_Position = gWorld * vec4(Position, 1.0); TexCoord0 = TexCoord; }";
//const char* pFS = "#version 330\n in vec2 TexCoord0; out vec4 FragColor; uniform sampler2D gSampler; void main() {FragColor = texture2D(gSampler, TexCoord0.xy);;}";
static const char* pVS = "                                                      \n\
    #version 330                                                                   \n\
    layout (location = 0) in vec3 pos;                                             \n\
    layout (location = 1) in vec2 tex;                                             \n\
    uniform mat4 gWorld;                                                           \n\
    out vec2 tex0;                                                                 \n\
    void main()                                                                    \n\
    {                                                                              \n\
        gl_Position = gWorld * vec4(pos, 1.0);                                     \n\
        tex0 = tex;                                                                \n\
    }";

static const char* pFS = "                                                         \n\
    #version 330                                                                    \n\
    in vec2 tex0;                                                                   \n\
    uniform sampler2D gSampler;                                                     \n\
    out vec4 fragcolor;                                                             \n\
    void main()                                                                     \n\
    {                                                                               \n\
        fragcolor = texture2D(gSampler, tex0.xy);                                   \n\
    }";

struct vertex {
	glm::vec3 fst;
	glm::vec2 snd;

	vertex(glm::vec3 inp1, glm::vec2 inp2) {
		fst = inp1;
		snd = inp2;
	}
};

static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType)
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

	GLint success;
	glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar InfoLog[1024];
		glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
		fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
		exit(1);
	}

	glAttachShader(ShaderProgram, ShaderObj);
}

static void CompileShaders(GLuint ShaderProgram)
{

	GLint success;

	glLinkProgram(ShaderProgram);

	glGetProgramiv(ShaderProgram, GL_VALIDATE_STATUS, &success);
	if (success == 0) {
		GLchar ErrorLog[1024];
		glGetProgramInfoLog(ShaderProgram, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
		exit(1);
	}

	glValidateProgram(ShaderProgram);

	glUseProgram(ShaderProgram);
	gWorldLocation = glGetUniformLocation(ShaderProgram, "gWorld");
	assert(gWorldLocation != 0xFFFFFFFF);
}

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
public:
	Pipeline()
	{
		m_scale = glm::vec3(1.0f, 1.0f, 1.0f);
		m_worldPos = glm::vec3(0.0f, 0.0f, 0.0f);
		m_rotateInfo = glm::vec3(0.0f, 0.0f, 0.0f);
	}

	void Scale(float ScaleX, float ScaleY, float ScaleZ)
	{
		m_scale.x = ScaleX;
		m_scale.y = ScaleY;
		m_scale.z = ScaleZ;
	}

	void WorldPos(float PosX, float PosY, float PosZ)
	{
		m_worldPos.x = PosX;
		m_worldPos.y = PosY;
		m_worldPos.z = PosZ;
	}

	void Rotate(float RotateX, float RotateY, float RotateZ)
	{
		m_rotateInfo.x = RotateX;
		m_rotateInfo.y = RotateY;
		m_rotateInfo.z = RotateZ;
	}
	void SetCamera(glm::vec3& Pos, glm::vec3& Target, glm::vec3& Up)
	{
		m_camera.Pos = Pos;
		m_camera.Target = Target;
		m_camera.Up = Up;
	}

	const glm::mat4x4* GetTrans()
	{
		glm::mat4x4 TranslationTrans(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			m_worldPos.x, m_worldPos.y, m_worldPos.z, 1.0f);
		glm::mat4x4 RotateTrans = RotMat(m_rotateInfo.x, m_rotateInfo.y, m_rotateInfo.z);
		glm::mat4x4 ScaleTrans(
			m_scale.x, 0.0f, 0.0f, 0.0f,
			0.0f, m_scale.x, 0.0f, 0.0f,
			0.0f, 0.0f, m_scale.x, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);
		glm::mat4x4 ProjectionMatrix = InitPerspectiveProj();
		glm::mat4x4 CamRotation = InitCameraTransform(m_camera.Target, m_camera.Up);
		glm::mat4x4 CamMove = InitCameraTranslation(m_camera.Pos.x, m_camera.Pos.y, m_camera.Pos.z);

		m_transformation = TranslationTrans * RotateTrans * ScaleTrans * CamMove * CamRotation * ProjectionMatrix;
		return &m_transformation;
	}
	glm::mat4x4 InitPerspectiveProj() const
	{
		const float ar = m_persProj.Width / m_persProj.Height;
		const float zNear = m_persProj.zNear;
		const float zFar = m_persProj.zFar;
		const float zRange = zNear - zFar;
		const float tanHalfFOV = tanf(glm::radians(m_persProj.FOV / 2.0));
		glm::mat4x4 ProjMat(
			1.0f / (tanHalfFOV * ar), 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f / tanHalfFOV, 0.0f, 0.0f,
			0.0f, 0.0f, (-zNear - zFar) / zRange, 2.0f * zFar * zNear / zRange,
			0.0f, 0.0f, 1.0f, 0.0f);
		return(ProjMat);
	}
	void SetPerspectiveProj(float FOV, float Width, float Height, float zNear, float zFar)
	{
		m_persProj.FOV = FOV;
		m_persProj.Width = Width;
		m_persProj.Height = Height;
		m_persProj.zNear = zNear;
		m_persProj.zFar = zFar;
	}
	glm::vec3 Cross(glm::vec3 v, glm::vec3 u) {

		float x = v[1] * u[2] - v[2] * u[1];
		float y = v[2] * u[0] - v[0] * u[2];
		float z = v[0] * u[1] - v[1] * u[0];
		glm::vec3 crossed = { x, y, z };
		return (crossed);
	}
	glm::vec3 Normalize(glm::vec3 v) {
		float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
		v.x /= len;
		v.y /= len;
		v.z /= len;
		glm::vec3 normalized = { v.x, v.y, v.z };
		return (normalized);
	}
	glm::mat4x4 InitCameraTransform(const glm::vec3& Target, const glm::vec3& Up)
	{
		glm::vec3 N = Target;
		N = Normalize(N);
		glm::vec3 U = Up;
		U = Normalize(U);
		U = Cross(U, Target);
		glm::vec3 V = Cross(N, U);
		glm::mat4x4 CameraRotate(
			U.x, U.y, U.z, 0.0f,
			V.x, V.y, V.z, 0.0f,
			N.x, N.y, N.z, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);
		return (CameraRotate);
	}
	glm::mat4x4 InitCameraTranslation(float x, float y, float z)
	{
		glm::mat4x4 cammov(
			1.0f, 0.0f, 0.0f, -x,
			0.0f, 1.0f, 0.0f, -y,
			0.0f, 0.0f, 1.0f, -z,
			0.0f, 0.0f, 0.0f, 1.0f);
		return (cammov);
	}
private:
	glm::vec3 m_scale;
	glm::vec3 m_worldPos;
	glm::vec3 m_rotateInfo;
	glm::mat4x4 m_transformation;
	glm::mat4x4 camera_move;
	glm::mat4x4 camera_rotate;
	struct {
		float FOV;
		float Width;
		float Height;
		float zNear;
		float zFar;
	} m_persProj;
	struct {
		glm::vec3 Pos;
		glm::vec3 Target;
		glm::vec3 Up;
	} m_camera;
};

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
void RenderSceneCB()
{
	glClear(GL_COLOR_BUFFER_BIT); //очистка буфера кадра
	
	scale += 0.015f;
	Pipeline p;
	//p.Scale(sinf(scale * 0.1f), sinf(scale * 0.1f), sinf(scale * 0.1f));
	//p.WorldPos(0.0f, 0.0f, sinf(scale));
	p.Rotate(sinf(scale) * 2, 0.0f, sinf(scale));
	p.SetPerspectiveProj(30.0f, GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT, 1.0f, 100.0f);

	glm::vec3 CameraPos(0.0f, 0.0f, -3.0f);
	glm::vec3 CameraTarget(0.0f, 0.0f, 2.0f);
	glm::vec3 CameraUp(0.0f, 1.0f, 0.0f);
	p.SetCamera(CameraPos, CameraTarget, CameraUp);
	
	glUniformMatrix4fv(gWorldLocation, 1, GL_TRUE, (const GLfloat*)p.GetTrans());

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), 0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (const GLvoid*)12);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	pTexture->Bind(GL_TEXTURE0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glutSwapBuffers(); // меняем фоновый буфер и буфер кадра местами
	glutIdleFunc(RenderSceneCB);

}
static void CreateVertexBuffer()
{
	vertex Vertices[4] = {
		vertex(glm::vec3(-0.2, -0.2, 0),glm::vec2(0,0)),
	   vertex(glm::vec3(0.3, -0.2, 0.5),glm::vec2(0.5,0)),
	   vertex(glm::vec3(0.3, -0.2, -0.5),glm::vec2(1,0)),
	   vertex(glm::vec3(0, 0.4, 0),glm::vec2(0.5,1)),
	};
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW);
}

static void CreateIndexBuffer()
{
	unsigned int Indices[] = { 0, 3, 1,
							   1, 3, 2,
							   2, 3, 0,
							   0, 2, 1 };

	glGenBuffers(1, &IBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices), Indices, GL_STATIC_DRAW);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv); // инициализация окна

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); //установка режима отображения
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT); //задаем размер, позицию окна, даем название
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Laba 3");

	GLenum res = glewInit();
	if (res != GLEW_OK)
	{
		fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
		return 1;
	}
	Magick::InitializeMagick(*argv);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	CreateVertexBuffer();
	CreateIndexBuffer();

	GLuint ShaderProgram = glCreateProgram();
	AddShader(ShaderProgram, pVS, GL_VERTEX_SHADER);
	AddShader(ShaderProgram, pFS, GL_FRAGMENT_SHADER);

	CompileShaders(ShaderProgram);
	glFrontFace(GL_CW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE); 
	
	glUniform1i(gSampler, 0);
	
	pTexture = new Texture(GL_TEXTURE_2D, "test.png");

	if (!pTexture->Load()) {
		return 1;
	}


	glutDisplayFunc(RenderSceneCB); //работа в самой оконной системе
	glutMainLoop(); //зацикливаем и вызываем функцию отображения окна на экран

	return 0;
}