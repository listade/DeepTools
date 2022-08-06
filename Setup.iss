#define AppName "DeepTools"
#define AppVersion "1.0"
#define AppPublisher "Sholukh Egor"

[Setup]
AppId={{7b281e88-150e-46f6-a11d-c46d70783e4e}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir="."

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "app\*"; DestDir: "{app}\app"; Flags: recursesubdirs ignoreversion 
Source: "cfg\*"; DestDir: "{app}\cfg"; Flags: recursesubdirs ignoreversion
Source: "data\*"; DestDir: "{app}\data"; Flags: recursesubdirs ignoreversion

[Run]
Filename: "{src}\build.bat"; Parameters: """{app}"""; Flags: waituntilterminated shellexec
Filename: "{cmd}"; Parameters: "/k cd ""{app}"" & env\Scripts\activate.bat"; Description: "Launch application"; Flags: nowait unchecked postinstall

[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"