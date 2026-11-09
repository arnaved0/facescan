// 3D preview of a processed scan (print space: millimetres, Z up, face towards -Y).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x26282c);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404048, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-300, -500, 400);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.6);
    rim.position.set(400, 300, 200);
    this.scene.add(rim);

    this.material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0 });
    this.mesh = null;
    this.grid = null;

    this._resize = () => this.resize();
    window.addEventListener("resize", this._resize);
    this.resize();
    const loop = () => {
      this._frame = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async load(url) {
    const geometry = await new PLYLoader().loadAsync(url);
    geometry.computeVertexNormals();
    this.clear();
    this.hasColor = !!geometry.getAttribute("color");
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.scene.add(this.mesh);

    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    const plate = Math.ceil((Math.max(size.x, size.y) * 1.6) / 10) * 10;
    this.grid = new THREE.GridHelper(plate, plate / 10, 0x666a72, 0x3a3d43);
    this.grid.rotation.x = Math.PI / 2; // grid in the XY plane (build plate)
    this.scene.add(this.grid);
    this.size = size;
    this.resetView();
  }

  resetView() {
    if (!this.size) return;
    const s = this.size;
    const target = new THREE.Vector3(0, 0, s.z * 0.5);
    const dist = Math.max(s.x, s.z) * 2.6;
    this.camera.position.set(dist * 0.35, -dist, s.z * 0.65);
    this.controls.target.copy(target);
    this.controls.update();
  }

  setColor(on) {
    this.material.vertexColors = on && this.hasColor;
    this.material.color.set(this.material.vertexColors ? 0xffffff : 0xd9d4cc);
    this.material.needsUpdate = true;
  }

  setWireframe(on) {
    this.material.wireframe = on;
  }

  clear() {
    for (const obj of [this.mesh, this.grid]) {
      if (obj) {
        this.scene.remove(obj);
        obj.geometry?.dispose();
      }
    }
    this.mesh = this.grid = null;
    this.size = null;
  }
}
