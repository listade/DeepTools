; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define AppName "DeepTools"
#define AppVersion "1.0"
#define AppPublisher "Sholukh Egor"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
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
