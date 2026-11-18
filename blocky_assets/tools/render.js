const {chromium}=require('playwright'); const fs=require('fs'); const path=require('path'); const http=require('http');
const [,,root,out]=process.argv;
const VIEW=__dirname;
const srv=http.createServer((q,s)=>{let p=decodeURIComponent(q.url.split('?')[0]); let f=p.startsWith('/assets/')?path.join(root,p.slice(8)):path.join(VIEW,p);
 fs.readFile(f,(e,d)=>{if(e){s.writeHead(404);return s.end();} s.writeHead(200,{'Content-Type':f.endsWith('.js')?'text/javascript':f.endsWith('.html')?'text/html':'application/octet-stream'}); s.end(d);});}).listen(8765);
(async()=>{const b=await chromium.launch({args:['--use-gl=swiftshader','--enable-webgl','--ignore-gpu-blocklist']}); const pg=await b.newPage();
 pg.on('console',m=>{if(m.type()==='error')console.log('ERR',m.text())});
 await pg.goto('http://localhost:8765/view.html'); await pg.waitForFunction('window.ready');
 const files=[]; (function walk(d){for(const f of fs.readdirSync(d)){const p=path.join(d,f); if(fs.statSync(p).isDirectory())walk(p); else if(f.endsWith('.glb'))files.push(path.relative(root,p));}})(path.join(root,'models'));
 fs.mkdirSync(out,{recursive:true});
 for(const f of files){ try{const url=await pg.evaluate(([u])=>window.render(u),['/assets/'+f]); fs.writeFileSync(path.join(out,path.basename(f,'.glb')+'.png'),Buffer.from(url.split(',')[1],'base64'));}catch(e){console.log('FAIL',f,e.message.slice(0,200));}}
 await b.close(); srv.close(); console.log('rendered',files.length);})();
