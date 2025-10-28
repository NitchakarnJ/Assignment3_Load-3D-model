// Grid Walk 3D — Models version (Top-only camera, polished walls/floor + fog)
// Needs: glad, glfw, glm, assimp, stb_image
// LearnOpenGL helpers: FileSystem, Shader (shader_m.h), Model

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>   // glm::pi, glm::two_pi

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/model.h>

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

// ---------------- Window ----------------
static const unsigned int GW_SCR_WIDTH = 800;
static const unsigned int GW_SCR_HEIGHT = 600;

// ---------------- Map ----------------
static std::vector<std::string> GW_MAP = {
"###############",
"#K....#.......#",
"#.###.#.#####.#",
"#.#...#.....#.#",
"#.#.#####.#.#.#",
"#.#.....#...#.#",
"#.#####.#.#.#.#",
"#.....#.#.#...#",
"###.#.#.#.###.#",
"#P..#.#.......#",
"#.###.#.#####.#",
"#.....#.....#.#",
"#.#####.###.#.#",
"#.......#..G#.#",
"###############"
};
// เก็บแผนที่ต้นฉบับไว้สำหรับรีเกม
static std::vector<std::string> GW_MAP_ORIG = GW_MAP;

static int GW_GRID_W = 0, GW_GRID_H = 0;

static void gw_fixMapWidth() {
    GW_GRID_H = (int)GW_MAP.size();
    GW_GRID_W = (int)GW_MAP[0].size();
    for (auto& r : GW_MAP) {
        if ((int)r.size() < GW_GRID_W) r += std::string(GW_GRID_W - (int)r.size(), '#');
        if ((int)r.size() > GW_GRID_W) r = r.substr(0, GW_GRID_W);
    }
}
static inline bool gw_wallAt(int x, int y) {
    if (x < 0 || x >= GW_GRID_W || y < 0 || y >= GW_GRID_H) return true;
    return GW_MAP[y][x] == '#';
}
static inline glm::vec2 gw_centerOf(const glm::ivec2& t) { return glm::vec2(t) + glm::vec2(0.5f); }

// ---------------- Minimal color shader (with fog) ----------------
static const char* GW_COLOR_VS = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;

uniform mat4 model, view, projection;

out vec3 N;
out vec3 Vpos;     // view-space position สำหรับคำนวณหมอก

void main() {
    mat3 Nmat = mat3(transpose(inverse(model)));
    N = normalize(Nmat * aNormal);

    vec4 worldPos = model * vec4(aPos, 1.0);
    vec4 viewPos  = view * worldPos;
    Vpos = viewPos.xyz;                 // เก็บใน view space

    gl_Position = projection * viewPos;
})";

static const char* GW_COLOR_FS = R"(#version 330 core
in vec3 N;
in vec3 Vpos;

out vec4 FragColor;

uniform vec3 uColor;

// ปรับแต่งบรรยากาศ
const vec3  fogColor   = vec3(0.04, 0.05, 0.08);  // สีฉากหลัง
const float fogDensity = 0.045;                   // เข้มหมอก

void main() {
    vec3 L = normalize(vec3(0.8, 1.2, 0.7));
    float d = max(dot(normalize(N), L), 0.0);
    vec3 base = uColor * (0.25 + 0.75 * d);

    // Fog แบบ exponential squared
    float dist = length(Vpos);
    float fog  = clamp(exp(-pow(fogDensity * dist, 2.0)), 0.0, 1.0);
    vec3 col   = mix(fogColor, base, fog);

    FragColor = vec4(col, 1.0);
})";

static GLuint gw_makeProgram(const char* vs, const char* fs) {
    auto comp = [&](GLenum t, const char* s) {
        GLuint id = glCreateShader(t);
        glShaderSource(id, 1, &s, nullptr); glCompileShader(id);
        GLint ok; glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[1024]; glGetShaderInfoLog(id, 1024, nullptr, log); std::cerr << log << "\n"; }
        return id;
        };
    GLuint v = comp(GL_VERTEX_SHADER, vs), f = comp(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f); return p;
}

static GLuint gw_colorProg = 0, gw_cubeVAO = 0, gw_cubeVBO = 0;
static void gw_initCube() {
    if (gw_cubeVAO) return;
    const float v[] = {
      -0.5f,0,-0.5f,0,0,-1,  0.5f,0,-0.5f,0,0,-1,  0.5f,1,-0.5f,0,0,-1,
      -0.5f,0,-0.5f,0,0,-1,  0.5f,1,-0.5f,0,0,-1, -0.5f,1,-0.5f,0,0,-1,
      -0.5f,0,0.5f ,0,0, 1,  0.5f,0,0.5f ,0,0, 1,  0.5f,1,0.5f ,0,0, 1,
      -0.5f,0,0.5f ,0,0, 1,  0.5f,1,0.5f ,0,0, 1, -0.5f,1,0.5f ,0,0, 1,
      -0.5f,0,-0.5f,-1,0,0,  -0.5f,0,0.5f,-1,0,0,  -0.5f,1,0.5f,-1,0,0,
      -0.5f,0,-0.5f,-1,0,0,  -0.5f,1,0.5f,-1,0,0,  -0.5f,1,-0.5f,-1,0,0,
       0.5f,0,-0.5f, 1,0,0,   0.5f,0,0.5f, 1,0,0,   0.5f,1,0.5f, 1,0,0,
       0.5f,0,-0.5f, 1,0,0,   0.5f,1,0.5f, 1,0,0,   0.5f,1,-0.5f, 1,0,0,
      -0.5f,1,-0.5f,0,1,0,    0.5f,1,-0.5f,0,1,0,   0.5f,1,0.5f,0,1,0,
      -0.5f,1,-0.5f,0,1,0,    0.5f,1,0.5f,0,1,0,   -0.5f,1,0.5f,0,1,0,
      -0.5f,0,-0.5f,0,-1,0,   0.5f,0,-0.5f,0,-1,0,  0.5f,0,0.5f,0,-1,0,
      -0.5f,0,-0.5f,0,-1,0,   0.5f,0,0.5f,0,-1,0,  -0.5f,0,0.5f,0,-1,0,
    };
    glGenVertexArrays(1, &gw_cubeVAO); glGenBuffers(1, &gw_cubeVBO);
    glBindVertexArray(gw_cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gw_cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

static void gw_drawCube(const glm::mat4& V, const glm::mat4& P, const glm::vec3& pos,
    const glm::vec3& size, const glm::vec3& color, float yawDeg = 0.f) {
    glUseProgram(gw_colorProg);
    glm::mat4 M(1.f);
    M = glm::translate(M, pos);
    M = glm::rotate(M, glm::radians(yawDeg), glm::vec3(0, 1, 0));
    M = glm::scale(M, size);
    glUniformMatrix4fv(glGetUniformLocation(gw_colorProg, "model"), 1, GL_FALSE, glm::value_ptr(M));
    glUniformMatrix4fv(glGetUniformLocation(gw_colorProg, "view"), 1, GL_FALSE, glm::value_ptr(V));
    glUniformMatrix4fv(glGetUniformLocation(gw_colorProg, "projection"), 1, GL_FALSE, glm::value_ptr(P));
    glUniform3f(glGetUniformLocation(gw_colorProg, "uColor"), color.x, color.y, color.z);
    glBindVertexArray(gw_cubeVAO); glDrawArrays(GL_TRIANGLES, 0, 36); glBindVertexArray(0);
}

// ==== Sphere mesh (for bullets / effects) ====
static GLuint gw_sphereVAO = 0, gw_sphereVBO = 0, gw_sphereEBO = 0;
static GLsizei gw_sphereIndexCount = 0;

static void gw_initSphere(int stacks = 12, int slices = 18) {
    if (gw_sphereVAO) return;

    std::vector<float> verts;   // interleaved: pos(3) + normal(3)
    std::vector<unsigned int> idx;
    verts.reserve((stacks + 1) * (slices + 1) * 6);

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / stacks;             // [0,1]
        float phi = v * glm::pi<float>();        // [0,pi]
        float cp = std::cos(phi), sp = std::sin(phi);

        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / slices;         // [0,1]
            float theta = u * glm::two_pi<float>();
            float ct = std::cos(theta), st = std::sin(theta);

            glm::vec3 n = { ct * sp, cp, st * sp }; // unit sphere
            verts.push_back(n.x); verts.push_back(n.y); verts.push_back(n.z); // pos
            verts.push_back(n.x); verts.push_back(n.y); verts.push_back(n.z); // normal
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int row1 = i * (slices + 1);
            int row2 = (i + 1) * (slices + 1);
            int a = row1 + j;
            int b = row1 + j + 1;
            int c = row2 + j;
            int d = row2 + j + 1;
            idx.push_back(a); idx.push_back(c); idx.push_back(b);
            idx.push_back(b); idx.push_back(c); idx.push_back(d);
        }
    }
    gw_sphereIndexCount = (GLsizei)idx.size();

    glGenVertexArrays(1, &gw_sphereVAO);
    glGenBuffers(1, &gw_sphereVBO);
    glGenBuffers(1, &gw_sphereEBO);

    glBindVertexArray(gw_sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gw_sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gw_sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned int), idx.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

static void gw_drawSphere(const glm::mat4& V, const glm::mat4& P,
    const glm::vec3& center, float radius,
    const glm::vec3& color) {
    glUseProgram(gw_colorProg);
    glm::mat4 M(1.0f);
    M = glm::translate(M, center);
    M = glm::scale(M, glm::vec3(radius)); // unit sphere -> radius

    glUniformMatrix4fv(glGetUniformLocation(gw_colorProg, "model"), 1, GL_FALSE, glm::value_ptr(M));
    glUniformMatrix4fv(glGetUniformLocation(gw_colorProg, "view"), 1, GL_FALSE, glm::value_ptr(V));
    glUniformMatrix4fv(glGetUniformLocation(gw_colorProg, "projection"), 1, GL_FALSE, glm::value_ptr(P));
    glUniform3f(glGetUniformLocation(gw_colorProg, "uColor"), color.x, color.y, color.z);

    glBindVertexArray(gw_sphereVAO);
    glDrawElements(GL_TRIANGLES, gw_sphereIndexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// ---------------- Entities ----------------
struct GWMoveCtrl {
    bool        moving = false;
    glm::ivec2  dir{ 0,0 };
    glm::ivec2  queued{ 0,0 };
    glm::vec2   target{ 0 };
};
struct GWEntity {
    glm::vec2 pos{ 0 };
    float     yaw = 0.f;
    GWMoveCtrl ctrl;
};
struct GWBullet {
    glm::vec2 pos;
    glm::vec2 dir;
    float     life = 1.5f;
    bool      alive = true;
};

// Speeds
static const float GW_STEP_SPEED_PLAYER = 6.0f;
static const float GW_STEP_SPEED_ENEMY = 5.0f;
static const float GW_BULLET_SPEED = 12.0f;
static const float GW_FIRE_COOLDOWN = 0.25f;

// Mouse wheel zoom
static float* GW_camDistPtr = nullptr;
static void gw_scroll_callback(GLFWwindow*, double, double yoffset) {
    if (!GW_camDistPtr) return;
    float& d = *GW_camDistPtr;
    d -= (float)yoffset * 0.8f;
    d = std::max(2.0f, std::min(30.0f, d));
}

// Input (tile direction)
static glm::ivec2 gw_readInput(GLFWwindow* w) {
    if (glfwGetKey(w, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) return { -1,0 };
    if (glfwGetKey(w, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) return { 1,0 };
    if (glfwGetKey(w, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) return { 0,-1 };
    if (glfwGetKey(w, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) return { 0, 1 };
    return { 0,0 };
}

// Enemy greedy steering
static glm::ivec2 gw_chooseDirChase(const glm::ivec2& fromTile, const glm::ivec2& curDir, const glm::ivec2& playerTile) {
    std::vector<glm::ivec2> dirs = { {1,0},{-1,0},{0,1},{0,-1} };
    glm::ivec2 best = curDir; int bestScore = 1e9;
    for (auto d : dirs) {
        if (d == -curDir) continue;
        glm::ivec2 nt = fromTile + d;
        if (gw_wallAt(nt.x, nt.y)) continue;
        int s = std::abs(playerTile.x - nt.x) + std::abs(playerTile.y - nt.y);
        if (s < bestScore) { bestScore = s; best = d; }
    }
    if (bestScore == 1e9) {
        glm::ivec2 rev = -curDir;
        if (!gw_wallAt(fromTile.x + rev.x, fromTile.y + rev.y)) return rev;
        return { 0,0 };
    }
    return best;
}

// Draw a Model with transforms
static void gw_drawModel(Shader& sh, Model& mdl, const glm::mat4& V, const glm::mat4& P,
    const glm::vec3& pos, const glm::vec3& scl = glm::vec3(1.0f),
    float yawDeg = 0.f, float pitchDeg = 0.f, float rollDeg = 0.f) {
    sh.use();
    glm::mat4 M(1.f);
    M = glm::translate(M, pos);
    if (yawDeg != 0.f)   M = glm::rotate(M, glm::radians(yawDeg), glm::vec3(0, 1, 0));
    if (pitchDeg != 0.f) M = glm::rotate(M, glm::radians(pitchDeg), glm::vec3(1, 0, 0));
    if (rollDeg != 0.f)  M = glm::rotate(M, glm::radians(rollDeg), glm::vec3(0, 0, 1));
    M = glm::scale(M, scl);
    sh.setMat4("model", M);
    sh.setMat4("view", V);
    sh.setMat4("projection", P);
    mdl.Draw(sh);
}

// ---------- Reset whole game state ----------
static void gw_resetGame(
    GWEntity& player, glm::vec2& playerSpawn,
    std::vector<GWEntity>& ghosts,
    std::vector<GWBullet>& bullets,
    bool& hasGun, float& fireCooldown
) {
    // คืนแผนที่ต้นฉบับ (มี 'K' กลับมา)
    GW_MAP = GW_MAP_ORIG;
    gw_fixMapWidth();

    // ล้างสถานะ
    ghosts.clear();
    bullets.clear();
    hasGun = false;
    fireCooldown = 0.0f;
    player = GWEntity{}; // reset movement/yaw

    // สแกนหาจุดเกิดใหม่ของผู้เล่นและผี
    for (int y = 0; y < GW_GRID_H; ++y) for (int x = 0; x < GW_GRID_W; ++x) {
        if (GW_MAP[y][x] == 'P') {
            player.pos = { x + 0.5f, y + 0.5f };
            playerSpawn = player.pos;
            GW_MAP[y][x] = '.'; // ลบอักษรออกจากแมพที่ใช้เรนเดอร์
        }
        if (GW_MAP[y][x] == 'G') {
            GWEntity g; g.pos = { x + 0.5f, y + 0.5f };
            ghosts.push_back(g);
            GW_MAP[y][x] = '.';
        }
    }
    if (ghosts.empty()) { GWEntity g; g.pos = { GW_GRID_W - 2.5f, GW_GRID_H - 2.5f }; ghosts.push_back(g); }
}

int main() {
    gw_fixMapWidth();

    // ตัวแปรสถานะหลัก
    GWEntity player; glm::vec2 playerSpawn{ 0 };
    std::vector<GWEntity> ghosts;
    bool hasGun = false; float fireCooldown = 0.0f; std::vector<GWBullet> bullets; bool prevSpace = false;

    // รีเกมครั้งแรก
    gw_resetGame(player, playerSpawn, ghosts, bullets, hasGun, fireCooldown);

    // --- GL init ---
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(GW_SCR_WIDTH, GW_SCR_HEIGHT, "Assignment3", nullptr, nullptr);
    if (!win) { std::cerr << "GLFW window fail\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "GLAD fail\n"; return -1; }
    glEnable(GL_DEPTH_TEST);

    gw_colorProg = gw_makeProgram(GW_COLOR_VS, GW_COLOR_FS);
    gw_initCube();
    gw_initSphere();

    // Model shader + models
    Shader modelShader("1.model_loading.vs", "1.model_loading.fs");
    Model duck(FileSystem::getPath("resources/objects/duck2/duck.obj"));
    Model rock(FileSystem::getPath("resources/objects/rock/rock.obj"));
    Model gun(FileSystem::getPath("resources/objects/gun/gun.obj"));
    Model& playerModel = duck;
    Model& ghostModel = rock;
    Model& gunModel = gun;

    // Camera (Top-only)
    float camPitch = -58.0f; // ค่าตั้งต้นปรับให้สูงขึ้นเล็กน้อย
    float camDist = glm::length(glm::vec2(5.0f, 7.0f));
    float camYaw = 180.0f + player.yaw;
    const float CAM_PITCH_MIN = -89.0f, CAM_PITCH_MAX = -10.0f;

    double lastMX = 0.0, lastMY = 0.0; bool rotating = false, rmbPrimed = false;
    const float SENS_X = 0.15f, SENS_Y = 0.15f;

    GW_camDistPtr = &camDist;
    glfwSetScrollCallback(win, gw_scroll_callback);

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime(); float dt = float(now - last); last = now;
        glfwPollEvents();

        // RMB orbit
        static const float DEADZONE = 2.0f;
        int rmb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT);
        if (rmb == GLFW_PRESS && !rotating) {
            rotating = true; rmbPrimed = true; glfwGetCursorPos(win, &lastMX, &lastMY);
#ifdef GLFW_RAW_MOUSE_MOTION
            if (glfwRawMouseMotionSupported()) glfwSetInputMode(win, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
#endif
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        if (rmb == GLFW_RELEASE && rotating) {
            rotating = false;
#ifdef GLFW_RAW_MOUSE_MOTION
            if (glfwRawMouseMotionSupported()) glfwSetInputMode(win, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
#endif
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        if (rotating) {
            double mx, my; glfwGetCursorPos(win, &mx, &my);
            if (rmbPrimed) { lastMX = mx; lastMY = my; rmbPrimed = false; }
            else {
                double dx = mx - lastMX, dy = my - lastMY; lastMX = mx; lastMY = my;
                if (std::abs(dx) > DEADZONE || std::abs(dy) > DEADZONE) {
                    camYaw += float(dx) * SENS_X;
                    camPitch -= float(dy) * SENS_Y;
                    camPitch = std::max(CAM_PITCH_MIN, std::min(CAM_PITCH_MAX, camPitch));
                    if (camYaw > 180.f) camYaw -= 360.f;
                    if (camYaw < -180.f) camYaw += 360.f;
                }
            }
        }

        // Player movement (tile-by-tile) + pre-turn
        {
            glm::ivec2 in = gw_readInput(win);
            if (in != glm::ivec2(0)) player.ctrl.queued = in;

            auto apply_dir = [&](const glm::ivec2& d) {
                player.ctrl.dir = d;
                if (d == glm::ivec2(1, 0))      player.yaw = 0.f;
                else if (d == glm::ivec2(-1, 0)) player.yaw = 180.f;
                else if (d == glm::ivec2(0, 1))  player.yaw = 90.f;
                else if (d == glm::ivec2(0, -1)) player.yaw = -90.f;
                };

            if (!player.ctrl.moving && player.ctrl.queued != glm::ivec2(0)) {
                glm::ivec2 t = { (int)std::floor(player.pos.x), (int)std::floor(player.pos.y) };
                glm::ivec2 nt = t + player.ctrl.queued;
                if (!gw_wallAt(nt.x, nt.y)) {
                    apply_dir(player.ctrl.queued);
                    player.ctrl.target = gw_centerOf(nt);
                    player.ctrl.moving = true;
                }
            }

            if (player.ctrl.moving && player.ctrl.queued != glm::ivec2(0) && player.ctrl.queued != player.ctrl.dir) {
                bool orthogonal = (player.ctrl.queued.x == 0 && player.ctrl.dir.x != 0) ||
                    (player.ctrl.queued.y == 0 && player.ctrl.dir.y != 0);
                if (orthogonal) {
                    glm::ivec2 t = { (int)std::floor(player.pos.x), (int)std::floor(player.pos.y) };
                    glm::vec2  center = gw_centerOf(t);

                    glm::ivec2 turnTo = t + player.ctrl.queued;
                    if (!gw_wallAt(turnTo.x, turnTo.y)) {
                        const float TURN_SNAP = 0.20f;
                        glm::vec2  toC = center - player.pos;
                        float      distC = glm::length(toC);

                        float step = GW_STEP_SPEED_PLAYER * dt;
                        bool willCrossCenter = (distC <= step + 1e-4f);

                        if (distC <= TURN_SNAP || willCrossCenter) {
                            player.pos = center;
                            apply_dir(player.ctrl.queued);
                            player.ctrl.target = gw_centerOf(turnTo);
                            player.ctrl.moving = true;
                        }
                    }
                }
            }

            if (player.ctrl.moving) {
                glm::vec2 to = player.ctrl.target - player.pos;
                float dist = glm::length(to);
                if (dist < 1e-4f) {
                    player.pos = player.ctrl.target;
                    player.ctrl.moving = false;
                }
                else {
                    glm::vec2 v = (to / std::max(dist, 1e-6f)) * GW_STEP_SPEED_PLAYER;
                    float step = GW_STEP_SPEED_PLAYER * dt;
                    if (step >= dist) { player.pos = player.ctrl.target; player.ctrl.moving = false; }
                    else player.pos += v * dt;
                }
            }

            // gun pickup
            glm::ivec2 pt = { (int)std::floor(player.pos.x), (int)std::floor(player.pos.y) };
            if (pt.x >= 0 && pt.x < GW_GRID_W && pt.y >= 0 && pt.y < GW_GRID_H && GW_MAP[pt.y][pt.x] == 'K') {
                hasGun = true; GW_MAP[pt.y][pt.x] = '.'; std::cout << "Picked up gun!\n";
            }
        }

        // Shooting
        {
            bool spaceNow = (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS);
            if (spaceNow && !prevSpace) {
                if (hasGun && fireCooldown <= 0.0f) {
                    glm::vec2 shootDir(0, -1);
                    if (player.ctrl.dir != glm::ivec2(0)) shootDir = glm::vec2((float)player.ctrl.dir.x, (float)player.ctrl.dir.y);
                    else {
                        if (fabs(player.yaw - 0.f) < 1e-3f)        shootDir = { 1, 0 };
                        else if (fabs(player.yaw - 180.f) < 1e-1f) shootDir = { -1, 0 };
                        else if (fabs(player.yaw - 90.f) < 1e-1f)  shootDir = { 0, 1 };
                        else if (fabs(player.yaw + 90.f) < 1e-1f)  shootDir = { 0,-1 };
                    }
                    if (glm::length(shootDir) > 0.0f) {
                        bullets.push_back(GWBullet{ player.pos, glm::normalize(shootDir), 1.5f, true });
                        fireCooldown = GW_FIRE_COOLDOWN;
                    }
                }
            }
            prevSpace = spaceNow;
        }

        // Ghosts
        glm::ivec2 playerTile = { (int)std::floor(player.pos.x), (int)std::floor(player.pos.y) };
        for (auto& g : ghosts) {
            glm::ivec2 gt = { (int)std::floor(g.pos.x), (int)std::floor(g.pos.y) };
            if (!g.ctrl.moving) {
                glm::ivec2 ndir = gw_chooseDirChase(gt, g.ctrl.dir, playerTile);
                if (ndir != glm::ivec2(0)) {
                    g.ctrl.dir = ndir;
                    g.ctrl.target = gw_centerOf(gt + ndir);
                    g.ctrl.moving = true;
                    if (g.ctrl.dir == glm::ivec2(1, 0))  g.yaw = 0.f;
                    else if (g.ctrl.dir == glm::ivec2(-1, 0)) g.yaw = 180.f;
                    else if (g.ctrl.dir == glm::ivec2(0, 1))  g.yaw = 90.f;
                    else if (g.ctrl.dir == glm::ivec2(0, -1)) g.yaw = -90.f;
                }
            }
            if (g.ctrl.moving) {
                glm::vec2 to = g.ctrl.target - g.pos;
                float dist = glm::length(to);
                if (dist < 1e-4f) { g.pos = g.ctrl.target; g.ctrl.moving = false; }
                else {
                    glm::vec2 v = (to / dist) * GW_STEP_SPEED_ENEMY;
                    float step = GW_STEP_SPEED_ENEMY * dt;
                    if (step >= dist) { g.pos = g.ctrl.target; g.ctrl.moving = false; }
                    else g.pos += v * dt;
                }
            }
        }

        // Bullets update
        for (auto& b : bullets) {
            if (!b.alive) continue;
            b.pos += b.dir * GW_BULLET_SPEED * dt;
            b.life -= dt;
            if (b.life <= 0.0f) b.alive = false;
            glm::ivec2 bt = { (int)std::floor(b.pos.x), (int)std::floor(b.pos.y) };
            if (bt.x < 0 || bt.x >= GW_GRID_W || bt.y < 0 || bt.y >= GW_GRID_H || gw_wallAt(bt.x, bt.y)) b.alive = false;
        }
        bullets.erase(std::remove_if(bullets.begin(), bullets.end(), [](const GWBullet& x) {return !x.alive; }), bullets.end());

        // Bullet vs Ghost
        for (auto itg = ghosts.begin(); itg != ghosts.end();) {
            bool killed = false;
            for (auto& b : bullets) {
                if (!b.alive) continue;
                if (glm::length(b.pos - itg->pos) < 0.7f) { b.alive = false; killed = true; break; }
            }
            if (killed) { std::cout << "Ghost shot!\n"; itg = ghosts.erase(itg); }
            else ++itg;
        }

        // Player vs Ghost — รีเกมทั้งกระดาน + ปืนเกิดใหม่
        bool collided = false;
        for (auto& g : ghosts) {
            if (glm::length(g.pos - player.pos) < 0.55f) { collided = true; break; }
        }
        if (collided) {
            std::cout << "Caught! Restart game.\n";
            gw_resetGame(player, playerSpawn, ghosts, bullets, hasGun, fireCooldown);
            // ไม่ยุ่งกับมุมกล้อง เพื่อไม่ให้เวียนหัวตอนรีเกม
        }

        // Camera
        glm::vec3 target = { player.pos.x, 0.7f, player.pos.y };
        float yawRad = glm::radians(camYaw), pitchRad = glm::radians(camPitch);
        glm::vec3 dir;
        dir.x = std::cos(pitchRad) * std::sin(yawRad);
        dir.y = std::sin(pitchRad);
        dir.z = std::cos(pitchRad) * std::cos(yawRad);
        glm::vec3 camPos = target - dir * camDist;

        glm::mat4 V = glm::lookAt(camPos, target, { 0,1,0 });
        glm::mat4 P = glm::perspective(glm::radians(55.f), (float)GW_SCR_WIDTH / (float)GW_SCR_HEIGHT, 0.1f, 200.f);

        // ===== Render =====
        glViewport(0, 0, GW_SCR_WIDTH, GW_SCR_HEIGHT);
        glClearColor(0.25f, 0.85f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ===== Floor (checkerboard) + Walls (alternate color) + Gun =====
        for (int y = 0; y < GW_GRID_H; ++y) {
            for (int x = 0; x < GW_GRID_W; ++x) {
                char c = GW_MAP[y][x];

                // พื้น: สร้างเป็น tile เตี้ยๆ (ความสูง 0.02) สีสลับแบบหมากรุก
                {
                    bool alt = ((x + y) & 1);
                    glm::vec3 floorColor = alt ? glm::vec3(0.08f, 0.09f, 0.13f)
                        : glm::vec3(0.10f, 0.12f, 0.16f);
                    gw_drawCube(V, P, { x + 0.5f, -0.01f, y + 0.5f }, { 1, 0.02f, 1 }, floorColor);
                }

                // ผนัง: สลับ 2 เฉดเพื่อให้เห็นทางชัดขึ้น
                if (c == '#') {
                    bool alt = ((x + y) & 1);
                    glm::vec3 wallColor = alt ? glm::vec3(0.12f, 0.35f, 0.85f)
                        : glm::vec3(0.10f, 0.30f, 0.76f);
                    gw_drawCube(V, P, { x + 0.5f, 0.5f, y + 0.5f }, { 1, 1, 1 }, wallColor);
                }
                else if (c == 'K') {
                    // ปืน: วางโมเดลไว้บนพื้น
                    glm::vec3 gp = { x + 0.5f, 0.15f, y + 0.5f };
                    gw_drawModel(modelShader, gunModel, V, P, gp, glm::vec3(0.0012f), 0.f, -90.f, 0.f);
                }
            }
        }

        // Player model (ปรับ yaw ให้หันถูกทิศ)
        float faceYaw = (player.ctrl.dir.y != 0) ? (player.yaw - 90.f) : (player.yaw + 90.f);
        gw_drawModel(modelShader, playerModel, V, P,
            { player.pos.x, 0.15f, player.pos.y },
            glm::vec3(1.0f),
            faceYaw, 0.f, 0.f);

        // Ghosts model — ปรับให้ rock ตั้งตรง/ไม่จม + เพิ่มแกนพลังด้านในให้เด่น
        const glm::vec3 GHOST_SCL = glm::vec3(0.35f); // ใหญ่ขึ้นจากเดิม
        const float     GHOST_Y = 0.25f;            // ยกจากพื้นเล็กน้อย
        const float     GHOST_PIT = 0.0f;             // ไม่ต้องก้ม

        for (auto& g : ghosts) {
            gw_drawModel(modelShader, ghostModel, V, P,
                { g.pos.x, GHOST_Y, g.pos.y },
                GHOST_SCL,
                g.yaw, GHOST_PIT, 0.f);

            // “แกนพลัง” สีเหลืองด้านใน
            gw_drawSphere(V, P, { g.pos.x, GHOST_Y + 0.10f, g.pos.y }, 0.10f, { 0.9f, 0.85f, 0.2f });
        }

        // Bullets — spheres
        for (auto& b : bullets)
            gw_drawSphere(V, P, { b.pos.x, 0.10f, b.pos.y }, 0.08f, { 1.0f, 0.95f, 0.2f });

        if (fireCooldown > 0.0f) fireCooldown -= dt;

        glfwSwapBuffers(win);
    }

    glfwTerminate();
    return 0;
}
