# 界面设计与开源来源

本次设计：石墨蓝、薄荷青、半透明磨砂、细边框、SVG 线性图标、等宽数据数字。Logo 为本项目原创 SVG（道路 + 检测框 + 视觉定位），源文件 ../logo.svg。

- Lucide：实际使用并内嵌的 SVG 图标，https://github.com/lucide-icons/lucide ，完整许可保存在 ../vendor/lucide/LICENSE。来源为 main，下载日期 2026-10-17。
- shadcn/ui Card：参考卡片、标题、内容与操作区的层级组织，未复制 React 代码或引入运行依赖。https://github.com/shadcn-ui/ui/blob/main/apps/v4/registry/new-york-v4/ui/card.tsx
- Tailwind CSS backdrop blur：参考透明背景结合 backdrop-filter 的磨砂实现。使用原生 CSS，无 Tailwind 依赖。https://tailwindcss.com/docs/backdrop-filter-blur

所有图标内嵌于 HTML，页面无 CDN 或外部字体请求。保留 reduced-motion / reduced-transparency 与 blur 不可用时的样式。

## Lucide 源文件 SHA-256

- activity.svg: 050fbcda05e13c2051cf857683645e19c6b180861ef4d0cc28a6f09ba1ea1666
- arrow-up-right.svg: 50b2503b9d11881142255466b7e3461d022b919735841c321d72003ac9959fe1
- camera.svg: ebbb3178dd80dfd942770a2237fe7bfc3515c55c7daa9d3666933a656d2790e3
- circle-check.svg: 9711e045a5999f8a3b0782eb88e09b04e0559bc7b379dcc0a4ae645223e10345
- circle-dot.svg: c263f6a08a089c06ad40f408ad5da9100359ab52d379e6c571cd7c5bd74eaa87
- cpu.svg: ec83bb69ec029d367d749afc445b39c8e95891ebf99b0400652677c2b149b99c
- crosshair.svg: ae9ed8583286ea03bffae85bb69aee5227540e588b0370cf185f2141ebe25e8d
- download.svg: 3daeee13ded5a3afa2198b1bb4e2262d163922f97d96ac6ff9953e8ed4fe039d
- gauge.svg: d7710543a5085c5a61d0bbac1da8b89ef28e541b38ed739357991dff33d475b5
- hard-drive.svg: 324363c40f54e4ca843de0ad1d023b063b70a26a83bb2e0a9e9f4aad9af37378
- image.svg: 619c9db58c879dce6250376ba67d9c9102ca6dbb3b1efb9294c9f180e22273a5
- layers.svg: ce16d0a955e3581cda17f6b2fd5c5bc2034b405f9cecc31c8ebb8495b4afc1bb
- memory-stick.svg: 9214439ac142c10156434542908436087fb174c71c46e30cfb46971c56ba33d2
- play.svg: d7c34786135922a92b6896f6c2384ceeb0346afbf6041dc79982011411409833
- scan-eye.svg: 343741486dfe4188a80628e3b8ae7a71f6ac63e0de5fbf474674b053c8f80721
- scan-line.svg: d17ec0ebae568a92e18e18968bb85658fec088bc3d97b92b729df6f7f9c4848e
- sliders-horizontal.svg: e43a00e5eb684e6cbb61083bed2528d7a3fa5f265693eb2b5f314cd432aa65f4
- square.svg: bd979354f0ab184b95cecf03eedefe40c2dc65830ac6d7e60017b2b25a354acb
- thermometer.svg: a56c95062445d52d95c51a77c4959d5a022539a0194205cc2ce2e9ce8f35847b
- upload.svg: 42816ee1fa0e9d3b82272a97615c1e562b0e9de64780ebf21963bc1db0ceffd1
- video.svg: d4fb6587f282c1152b953fe9494cd8ccafdf1e9f44957fbb7fe8dc7af0457597

页面默认采用用户明确要求的磨砂材质，顶栏提供「磨砂 / 实色」切换。显式选择优先于系统默认透明度偏好；浏览器不支持 blur 时仍使用实色。
