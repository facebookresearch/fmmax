"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[9688],{3905:(e,t,r)=>{r.d(t,{Zo:()=>p,kt:()=>b});var a=r(7294);function n(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,a)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){n(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,a,n=function(e,t){if(null==e)return{};var r,a,n={},i=Object.keys(e);for(a=0;a<i.length;a++)r=i[a],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)r=i[a],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var c=a.createContext({}),l=function(e){var t=a.useContext(c),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},p=function(e){var t=l(e.components);return a.createElement(c.Provider,{value:t},e.children)},u="mdxType",f={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},m=a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,i=e.originalType,c=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),u=l(r),m=n,b=u["".concat(c,".").concat(m)]||u[m]||f[m]||i;return r?a.createElement(b,o(o({ref:t},p),{},{components:r})):a.createElement(b,o({ref:t},p))}));function b(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=r.length,o=new Array(i);o[0]=m;var s={};for(var c in t)hasOwnProperty.call(t,c)&&(s[c]=t[c]);s.originalType=e,s[u]="string"==typeof e?e:n,o[1]=s;for(var l=2;l<i;l++)o[l]=r[l];return a.createElement.apply(null,o)}return a.createElement.apply(null,r)}m.displayName="MDXCreateElement"},5284:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>c,contentTitle:()=>o,default:()=>f,frontMatter:()=>i,metadata:()=>s,toc:()=>l});var a=r(7462),n=(r(7294),r(3905));const i={id:"basis.LatticeVectors"},o=void 0,s={unversionedId:"API/basis.LatticeVectors",id:"API/basis.LatticeVectors",title:"basis.LatticeVectors",description:"Class basis.LatticeVectors():",source:"@site/docs/API/basis.LatticeVectors.md",sourceDirName:"API",slug:"/API/basis.LatticeVectors",permalink:"/fmmax/API/basis.LatticeVectors",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/basis.LatticeVectors.md",tags:[],version:"current",frontMatter:{id:"basis.LatticeVectors"},sidebar:"APISidebar",previous:{title:"basis.Expansion",permalink:"/fmmax/API/basis.Expansion"},next:{title:"basis.Truncation",permalink:"/fmmax/API/basis.Truncation"}},c={},l=[{value:"<code>Class basis.LatticeVectors():</code>",id:"class-basislatticevectors",level:3},{value:"Args:",id:"args",level:4}],p={toc:l},u="wrapper";function f(e){let{components:t,...r}=e;return(0,n.kt)(u,(0,a.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h3",{id:"class-basislatticevectors"},(0,n.kt)("inlineCode",{parentName:"h3"},"Class basis.LatticeVectors():")),(0,n.kt)("p",null,"Stores a pair of lattice vectors."),(0,n.kt)("p",null,"Note that this is just a pair of 2-dimensional vectors, which may be either for\nthe real-space lattice or the reciprocal space lattice, depending on usage."),(0,n.kt)("h4",{id:"args"},"Args:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"u"),": The first primitive lattice vector."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"v"),": The second primitive lattice vector, with identical shape.")))}f.isMDXComponent=!0}}]);