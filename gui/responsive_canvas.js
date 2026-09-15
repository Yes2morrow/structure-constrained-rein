// Fit the complete Fabric surface and toolbar inside this iframe, then allow
// view-only mouse zooming. CSS scaling leaves serialized geometry untouched;
// Fabric maps pointer pixels using the canvas bounding rectangle.
(() => {
  let width=680, height=440, pending=false, baseScale=1, viewZoom=1;
  let panLeft=0, panTop=0, panning=false, panX=0, panY=0, startLeft=0, startTop=0;

  function ensureControls(stage) {
    if (!stage || document.getElementById('environment-view-controls')) return;
    const controls=document.createElement('div');
    controls.id='environment-view-controls';
    controls.innerHTML='<button type="button" data-action="out" title="缩小">−</button><span>100%</span><button type="button" data-action="in" title="放大">＋</button><button type="button" data-action="reset" title="恢复完整视图">重置</button>';
    Object.assign(controls.style,{
      display:'flex',justifyContent:'flex-end',alignItems:'center',gap:'4px',
      padding:'4px 6px',border:'1px solid #cbd5e1',borderRadius:'9px',
      background:'rgba(255,255,255,.94)',boxSizing:'border-box',
      font:'12px sans-serif',color:'#334155'
    });
    controls.querySelectorAll('button').forEach(button=>Object.assign(button.style,{
      border:'1px solid #cbd5e1',borderRadius:'6px',background:'#fff',cursor:'pointer',
      minWidth:'28px',height:'25px',padding:'0 7px',color:'#1e293b'
    }));
    controls.addEventListener('pointerdown',event=>event.stopPropagation());
    controls.addEventListener('click',event=>{
      const action=event.target?.dataset?.action;
      if(action==='in') setZoom(viewZoom*1.15);
      if(action==='out') setZoom(viewZoom/1.15);
      if(action==='reset') setZoom(1,true);
    });
    stage.appendChild(controls);
  }

  function fit() {
    pending=false;
    const root=document.getElementById('root');
    if (!root) return;
    baseScale=Math.max(.05,document.documentElement.clientWidth/width);
    root.style.width=width+'px';
    root.style.transformOrigin='top left';
    root.style.transform='scale('+baseScale+')';
    const stage=root.firstElementChild;
    ensureControls(stage);
    if(stage) {
      const layers=Array.from(stage.children);
      layers.forEach((layer,index)=>{
        const isCanvasLayer=index<3;
        layer.style.transformOrigin='top left';
        layer.style.transform=isCanvasLayer
          ? 'translate('+panLeft+'px,'+panTop+'px) scale('+viewZoom+')'
          : '';
        if(!isCanvasLayer) {
          layer.style.background='#ffffff';
          layer.style.width=width+'px';
          layer.style.minHeight='36px';
        }
      });
    }
    document.body.style.margin='0';
    document.body.style.overflow='hidden';
    document.body.style.width='100vw';
    document.body.style.height=Math.ceil((height+80)*baseScale)+'px';
    const label=document.querySelector('#environment-view-controls span');
    if(label) label.textContent=Math.round(viewZoom*100)+'%';
    window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:Math.ceil((height+80)*baseScale)},'*');
  }
  function schedule(){if(!pending){pending=true;requestAnimationFrame(fit);}}
  function setZoom(next, resetView=false, clientX=null, clientY=null) {
    const anchorClientX=clientX??document.documentElement.clientWidth/2;
    const anchorClientY=clientY??document.documentElement.clientHeight/2;
    const anchorX=(anchorClientX/baseScale-panLeft)/viewZoom;
    const anchorY=(anchorClientY/baseScale-panTop)/viewZoom;
    viewZoom=Math.max(.5,Math.min(3,next));
    if(resetView) {
      panLeft=0;
      panTop=0;
    } else {
      panLeft=anchorClientX/baseScale-anchorX*viewZoom;
      panTop=anchorClientY/baseScale-anchorY*viewZoom;
    }
    fit();
  }
  window.addEventListener('message',event=>{
    if(event.data?.type==='streamlit:render'){
      width=event.data.args.canvasWidth; height=event.data.args.canvasHeight;
      schedule(); setTimeout(schedule,150);
    }
  });
  new ResizeObserver(schedule).observe(document.documentElement);
  new MutationObserver(schedule).observe(document.getElementById('root'),{childList:true,subtree:true});
  window.addEventListener('resize',schedule);
  document.addEventListener('wheel',event=>{
    if(!event.target?.closest?.('canvas')) return;
    event.preventDefault();
    setZoom(viewZoom*(event.deltaY<0?1.1:1/1.1),false,event.clientX,event.clientY);
  },{passive:false,capture:true});
  document.addEventListener('pointerdown',event=>{
    if(!event.target?.closest?.('canvas') || !(event.button===1 || event.altKey)) return;
    panning=true; panX=event.clientX; panY=event.clientY;
    startLeft=panLeft; startTop=panTop;
    event.preventDefault(); event.stopImmediatePropagation();
  },true);
  window.addEventListener('pointermove',event=>{
    if(!panning) return;
    panLeft=startLeft+(event.clientX-panX)/baseScale;
    panTop=startTop+(event.clientY-panY)/baseScale;
    fit();
    event.preventDefault();
  },true);
  window.addEventListener('pointerup',()=>{panning=false;},true);
  schedule();
})();
